# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
Solver-agnostic logic to compose the requests passed to the solver
and accumulate its results.

The state exposed to the solver is handled by two objects whose primary
function is to serve read-only information to the solver and its other helpers.

- ``SolverInputState``: fully solver agnostic. It handles:
    - The local state on disk, namely the prefix state. This includes the
      already installed packages in the prefix (if any), the explicit requests
      made in that prefix in the past (history), its pinned specs, packages
      configured as aggressive updates and others.
    - The runtime context, determined by the configuration file(s),
      `CONDA_*` environment variables, command line flags and the requested
      specs (if any).
- ``IndexHelper``: can be subclassed to add solver-specific logic
  (e.g. custom index building). It should, provide, at least, a method to
  query the index for the _explicit pool_ of packages for a given spec (e.g.
  its potential dependency tree). Note that the IndexHelper might need
  pieces of ``SolverInputState`` to build the index (e.g. installed packages,
  configured channels and subdirs...)

.. todo::

    Embed IndexHelper in SolverInputState?

Since ``conda`` follows an iterative approach to solve a request,
in addition the _input_ state, the Solver itself can store additional state
in a separate helper: the ``SolverOutputState`` object. This is meant to help
accumulate the following pieces of data:

- ``specs``: a mapping of package names to its corresponding ``MatchSpec``
  objects. These objects are passed to the actual Solver, hoping it will return
  a solution.
- ``records``: a mapping of package names to ``PackageRecord`` objects. It will
  end up containing the list of package records that will compose the final state
  of the prefix (the _solution_). Its default value is set to the currently installed
  packages in the prefix. The solver will alter this list as needed to accommodate
  the final solution.

If the algorithm was not iterative, the sole purpose of the solver would be to turn
the ``specs`` into ``records``. However, ``conda``'s logic will try to constrain the
solution to mimic the initial state as much as possible to reduce the amount of
changes in the prefix. Sometimes, the initial request is too constrained, which results
in a number of conflicts. These conflicts are then stored in the ``conflicts`` mapping,
which will determine which ``specs`` are relaxed in the next attempt. Additionally,
``conda`` stores other solve artifacts:

- ``for_history``: The explicitly requested specs in the command-line should end up
  in the history. Some modifier flags can affect how this mapping is populated (e.g.
  ``--update-deps``.)
- ``neutered``: Pieces of history that were found to be conflicting in the future and
  were annotated as such to avoid falling in the same conflict again.

The mappings stored in ``SolverOutputState`` are backed by ``TrackedMap`` objects,
which allow to keep the reasons _why_ those specs or records were added to the mappings,
as well as richer logging for each action.
"""

# TODO: This module could be part of conda-core once if we refactor the classic logic

import logging
from os import PathLike
from types import MappingProxyType
from typing import Iterable, Mapping, Optional, Type, Union

from boltons.setutils import IndexedSet
from conda import CondaError
from conda.auxlib import NULL
from conda.auxlib.ish import dals
from conda.base.constants import DepsModifier, UpdateModifier
from conda.base.context import context
from conda.common.io import dashlist
from conda.common.path import get_major_minor_version, paths_equal
from conda.core.index import _supplement_index_with_system
from conda.core.prefix_data import PrefixData
from conda.core.solve import get_pinned_specs
from conda.exceptions import PackagesNotFoundError, SpecsConfigurationConflictError
from conda.history import History
from conda.models.channel import Channel
from conda.models.match_spec import MatchSpec
from conda.models.prefix_graph import PrefixGraph
from conda.models.records import PackageRecord

from .models import EnumAsBools, TrackedMap
from .utils import compatible_specs

log = logging.getLogger(f"conda.{__name__}")


class IndexHelper:
    """
    The _index_ refers to the combination of all configured channels and their
    platform-corresponding subdirectories. It provides the sources for available
    packages that can become part of a prefix state, eventually.

    Subclass this helper to add custom repodata fetching if needed.
    """

    def explicit_pool(self, specs: Iterable[MatchSpec]) -> Iterable[str]:
        raise NotImplementedError


class SolverInputState:
    """
    Helper object to provide the input data needed to compute the state that will be
    exposed to the solver.

    Parameters
    ----------
    prefix
        Path to the prefix we are operating on. This will be used to expose
        ``PrefixData``, ``History``, pinned specs, among others.
    requested
        The MatchSpec objects required by the user (either in the command line or
        through the Python API).
    update_modifier
        A value of ``UpdateModifier``, which has an effect on which specs are added
        to the final list. The default value here must match the default value in the
        ``context`` object.
    deps_modifier
        A value of ``DepsModifier``, which has an effect on which specs are added
        to the final list. The default value here must match the default value in the
        ``context`` object.
    ignore_pinned
        Whether pinned specs can be ignored or not. The default value here must match
        the default value in the ``context`` object.
    force_remove
        Remove the specs without solving the environment (which would also remove their)
        dependencies. The default value here must match the default value in the
        ``context`` object.
    force_reinstall
        Uninstall and install the computed records even if they were already satisfied
        in the given prefix. The default value here must match the default value in the
        ``context`` object.
    prune
        Remove dangling dependencies that ended up orphan. The default value here must
        match the default value in the ``context`` object.
    command
        The subcommand used to invoke this operation (e.g. ``create``, ``install``, ``remove``...).
        It can have an effect on the computed list of records.
    _pip_interop_enabled
        Internal only. Whether ``PrefixData`` will also expose packages not installed by
        ``conda`` (e.g. ``pip`` and others can put Python packages in the prefix).
    """

    _ENUM_STR_MAP = {
        "NOT_SET": DepsModifier.NOT_SET,
        "NO_DEPS": DepsModifier.NO_DEPS,
        "ONLY_DEPS": DepsModifier.ONLY_DEPS,
        "SPECS_SATISFIED_SKIP_SOLVE": UpdateModifier.SPECS_SATISFIED_SKIP_SOLVE,
        "FREEZE_INSTALLED": UpdateModifier.FREEZE_INSTALLED,
        "UPDATE_DEPS": UpdateModifier.UPDATE_DEPS,
        "UPDATE_SPECS": UpdateModifier.UPDATE_SPECS,
        "UPDATE_ALL": UpdateModifier.UPDATE_ALL,
    }
    _DO_NOT_REMOVE_NAMES = (
        "anaconda",
        "conda",
        "conda-build",
        "python.app",
        "console_shortcut",
        "powershell_shortcut",
    )

    def __init__(
        self,
        prefix: Union[str, bytes, PathLike],
        requested: Optional[Iterable[Union[str, MatchSpec]]] = (),
        update_modifier: Optional[UpdateModifier] = UpdateModifier.UPDATE_SPECS,
        deps_modifier: Optional[DepsModifier] = DepsModifier.NOT_SET,
        ignore_pinned: Optional[bool] = None,
        force_remove: Optional[bool] = False,
        force_reinstall: Optional[bool] = False,
        prune: Optional[bool] = False,
        command: Optional[str] = None,
        _pip_interop_enabled: Optional[bool] = None,
    ):
        self.prefix = prefix
        self._prefix_data = PrefixData(prefix, pip_interop_enabled=_pip_interop_enabled)
        self._pip_interop_enabled = _pip_interop_enabled
        self._history = History(prefix).get_requested_specs_map()
        self._pinned = {spec.name: spec for spec in get_pinned_specs(prefix)}
        self._aggressive_updates = {spec.name: spec for spec in context.aggressive_update_packages}

        virtual = {}
        _supplement_index_with_system(virtual)
        self._virtual = {record.name: record for record in virtual}

        self._requested = {}
        for spec in requested:
            spec = MatchSpec(spec)
            self._requested[spec.name] = spec

        self._update_modifier = self._default_to_context_if_null(
            "update_modifier", update_modifier
        )
        if prune and self._update_modifier == UpdateModifier.FREEZE_INSTALLED:
            self._update_modifier = UpdateModifier.UPDATE_SPECS  # revert to default
        self._deps_modifier = self._default_to_context_if_null("deps_modifier", deps_modifier)
        self._ignore_pinned = self._default_to_context_if_null("ignore_pinned", ignore_pinned)
        self._force_remove = self._default_to_context_if_null("force_remove", force_remove)
        self._force_reinstall = self._default_to_context_if_null(
            "force_reinstall", force_reinstall
        )
        self._prune = prune
        self._command = command

        # special cases
        self._do_not_remove = {p: MatchSpec(p) for p in self._DO_NOT_REMOVE_NAMES}

    def _default_to_context_if_null(self, name, value, context=context):
        "Obtain default value from the context if value is set to NULL; otherwise leave as is"
        return getattr(context, name) if value is NULL else self._ENUM_STR_MAP.get(value, value)

    @property
    def prefix_data(self) -> PrefixData:
        """
        A direct reference to the ``PrefixData`` object for the given ``prefix``.
        You will usually use this object through the ``installed`` property.
        """
        return self._prefix_data

    # Prefix state pools

    @property
    def installed(self) -> Mapping[str, PackageRecord]:
        """
        This exposes the installed packages in the prefix. Note that a ``PackageRecord``
        can generate an equivalent ``MatchSpec`` object with ``.to_match_spec()``.
        Records are toposorted.
        """
        return MappingProxyType(dict(sorted(self.prefix_data._prefix_records.items())))

    @property
    def history(self) -> Mapping[str, MatchSpec]:
        """
        These are the specs that the user explicitly asked for in previous operations
        on the prefix. See :class:`History` for more details.
        """
        return MappingProxyType(self._history)

    @property
    def pinned(self) -> Mapping[str, MatchSpec]:
        """
        These specs represent hard constrains on what package versions can be installed
        on the environment. The packages here returned don't need to be already installed.

        If ``ignore_pinned`` is True, this returns an empty dictionary.
        """
        if self.ignore_pinned:
            return MappingProxyType({})
        return MappingProxyType(self._pinned)

    @property
    def virtual(self) -> Mapping[str, MatchSpec]:
        """
        System properties exposed as virtual packages (e.g. ``__glibc=2.17``). These packages
        cannot be (un)installed, they only represent constrains for other packages. By convention,
        their names start with a double underscore.
        """
        return MappingProxyType(dict(sorted(self._virtual.items())))

    @property
    def aggressive_updates(self) -> Mapping[str, MatchSpec]:
        """
        Packages that the solver will always try to update. As such, they will never have an
        associated version or build constrain. Note that the packages here returned do not need to
        be installed.
        """
        return MappingProxyType(self._aggressive_updates)

    @property
    def always_update(self) -> Mapping[str, MatchSpec]:
        """
        Merged lists of packages that should always be updated, depending on the flags, including:
        - aggressive_updates
        - conda if auto_update_conda is true and we are on the base env
        - almost all packages if update_all is true
        - etc
        """
        installed = self.installed
        pinned = self.pinned
        pkgs = {pkg: MatchSpec(pkg) for pkg in self.aggressive_updates if pkg in installed}
        if context.auto_update_conda and paths_equal(self.prefix, context.root_prefix):
            pkgs.setdefault("conda", MatchSpec("conda"))
        if self.update_modifier.UPDATE_ALL:
            for pkg in installed:
                if pkg != "python" and pkg not in pinned:
                    pkgs.setdefault(pkg, MatchSpec(pkg))
        return MappingProxyType(pkgs)

    @property
    def do_not_remove(self) -> Mapping[str, MatchSpec]:
        """
        Packages that are protected by the solver so they are not accidentally removed. This list
        is not configurable, but hardcoded for legacy reasons.
        """
        return MappingProxyType(self._do_not_remove)

    @property
    def requested(self) -> Mapping[str, MatchSpec]:
        """
        Packages that the user has explicitly asked for in this operation.
        """
        return MappingProxyType(self._requested)

    # Types of commands

    @property
    def is_installing(self) -> bool:
        """
        True if the used subcommand was ``install``.
        """
        return self._command == "install"

    @property
    def is_updating(self) -> bool:
        """
        True if the used subcommand was ``update``.
        """
        return self._command == "update"

    @property
    def is_creating(self) -> bool:
        """
        True if the used subcommand was ``create``.
        """
        return self._command == "create"

    @property
    def is_removing(self) -> bool:
        """
        True if the used subcommand was ``remove``.
        """
        return self._command == "remove"

    # modifiers

    @property
    def update_modifier(self) -> EnumAsBools:
        """
        Use attribute access to test whether the modifier is set to that value

        >>> update_modifier = EnumAsBools(context.update_modifier)
        >>> update_modifier.UPDATE_SPECS
            True
        >>> update_modifier.UPDATE_DEPS
            False
        """
        return EnumAsBools(self._update_modifier)

    @property
    def deps_modifier(self) -> EnumAsBools:
        """
        Use attribute access to test whether the modifier is set to that value

        >>> deps_modifier = EnumAsBools(context.deps_modifier)
        >>> deps_modifier.NOT_SET
            True
        >>> deps_modifier.DEPS_ONLY
            False
        """
        return EnumAsBools(self._deps_modifier)

    # Other flags

    @property
    def ignore_pinned(self) -> bool:
        return self._ignore_pinned

    @property
    def force_remove(self) -> bool:
        return self._force_remove

    @property
    def force_reinstall(self) -> bool:
        return self._force_reinstall

    @property
    def prune(self) -> bool:
        return self._prune

    #  Utility methods

    def channels_from_specs(self) -> Iterable[Channel]:
        """
        Collect all channels added with the `channel::package=*` syntax. For now,
        we only collect those specifically requested by the user in the current command
        (same as conda), but we should investigate whether history keeps channels around
        too.
        """
        for spec in self.requested.values():
            channel = spec.get_exact_value("channel")
            if channel:
                if spec.original_spec_str and spec.original_spec_str.startswith("file://"):
                    # Handle MatchSpec roundtrip issue with local channels
                    channel = Channel(spec.original_spec_str.split("::")[0])
                yield channel

    def channels_from_installed(self, seen=None) -> Iterable[Channel]:
        seen_urls = set(seen or [])
        # See https://github.com/conda/conda/issues/11790
        for record in self.installed.values():
            if record.channel.auth or record.channel.token:
                # skip if the channel has authentication info, because
                # it might cause issues with expired tokens and what not
                continue
            if record.channel.name in ("@", "<develop>", "pypi"):
                # These "channels" are not really channels, more like
                # metadata placeholders
                continue
            if record.channel.base_url is None:
                continue
            if record.channel.subdir_url in seen_urls:
                continue
            seen_urls.add(record.channel.subdir_url)
            yield record.channel

    def maybe_free_channel(self) -> Iterable[Channel]:
        if context.restore_free_channel:
            yield Channel.from_url("https://repo.anaconda.com/pkgs/free")


class SolverOutputState:
    # TODO: This object is starting to look _a lot_ like conda.core.solve itself...
    # Consider merging this with a base class in conda.core.solve
    """
    This is the main mutable object we will massage before passing the result of the computation
    (the ``specs`` mapping) to the solver. It will also store the result of the solve (in
    ``records``).

    Parameters
    ----------
    solver_input_state
        This instance provides the initial state for the output.
    specs
        Mapping of package names to ``MatchSpec`` objects that will override the initialization
        driven by ``solver_input_state`` (check ``._initialize_specs_from_input_state()`` for more
        details).
    records
        Mapping of package names to ``PackageRecord`` objects. If not provided, it will be
        initialized from the ``installed`` records in ``solver_input_state``.
    for_history
        Mapping of package names to ``MatchSpec`` objects. These specs will be written to
        the prefix history once the solve is complete. Its default initial value is taken from the
        explicitly requested packages in the ``solver_input_state`` instance.
    neutered
        Mapping of package names to ``MatchSpec`` objects. These specs are also written to
        the prefix history, as part of the neutered specs. If not provided, their default value is
        a blank mapping.
    conflicts
        If a solve attempt is not successful, conflicting specs are kept here for further
        relaxation of the version and build constrains. If not provided, their default value is a
        blank mapping.
    pins
        Packages that ended up being pinned. Mostly used for reporting and debugging.

    Notes
    -----
    Almost all the attributes in this object map package names (``str``) to ``MatchSpec``
    (_specs_ in short) objects. The only mapping with different values is ``records``, which
    stores ``PackageRecord`` objects. A quick note on these objects:

    * ``MatchSpec`` objects are a query language for packages, based on the ``PackageRecord``
      schema. ``PackageRecord`` objects is how packages that are already installed are
      represented. This is what you get from ``PrefixData.iter_records()``. Since they are
      related, ``MatchSpec`` objects can be created from a ``PackageRecord`` with
      ``.to_match_spec()``.
    * ``MatchSpec`` objects also feature fields like ``target`` and ``optional``. These are,
      essentially, used by the low-level classic solver (:class:`conda.resolve.Resolve`) to
      mark specs as items it can optionally play with to satisfy the solver constrains. A
      ``target`` marked spec is _soft-pinned_ in the sense that the solver will try to satisfy
      that but it will stop trying if it gets in the way, so you might end up a different
      version or build. ``optional`` seems to be in the same lines, but maybe the entire spec
      can be dropped from the request? The key idea here is that these two fields might not be
      directly usable by the solver, but it might need some custom adaptation. For example, for
      ``libmamba`` we might need a separate pool that can be configured as a flexible task. See
      more details in the first comment of ``conda.core.solve.classic.Solver._add_specs``
    """

    def __init__(
        self,
        *,
        solver_input_state: SolverInputState,
        specs: Optional[Mapping[str, MatchSpec]] = None,
        records: Optional[Mapping[str, PackageRecord]] = None,
        for_history: Optional[Mapping[str, MatchSpec]] = None,
        neutered: Optional[Mapping[str, MatchSpec]] = None,
        conflicts: Optional[Mapping[str, MatchSpec]] = None,
        pins: Optional[Mapping[str, MatchSpec]] = None,
    ):
        self.solver_input_state: SolverInputState = solver_input_state

        self.records: Mapping[str, PackageRecord] = TrackedMap("records")
        if records:
            self.records.update(records, reason="Initialized from explicitly passed arguments")
        elif solver_input_state.installed:
            self.records.update(
                solver_input_state.installed,
                reason="Initialized from installed packages in prefix",
            )

        self.specs: Mapping[str, MatchSpec] = TrackedMap("specs")
        if specs:
            self.specs.update(specs, reason="Initialized from explicitly passed arguments")
        else:
            self._initialize_specs_from_input_state()

        self.for_history: Mapping[str, MatchSpec] = TrackedMap("for_history")
        if for_history:
            self.for_history.update(
                for_history, reason="Initialized from explicitly passed arguments"
            )
        elif solver_input_state.requested:
            self.for_history.update(
                solver_input_state.requested,
                reason="Initialized from requested specs in solver input state",
            )

        self.neutered: Mapping[str, MatchSpec] = TrackedMap(
            "neutered", data=(neutered or {}), reason="From arguments"
        )

        # we track conflicts to relax some constrains and help the solver out
        self.conflicts: Mapping[str, MatchSpec] = TrackedMap(
            "conflicts", data=(conflicts or {}), reason="From arguments"
        )

        self.pins: Mapping[str, MatchSpec] = TrackedMap(
            "pins", data=(pins or {}), reason="From arguments"
        )

    def _initialize_specs_from_input_state(self):
        """
        Provide the initial value for the ``.specs`` mapping. This depends on whether
        there's a history available (existing prefix) or not (new prefix).
        """
        # Initialize specs following conda.core.solve._collect_all_metadata()

        if self.solver_input_state.prune:
            pass  # we do not initialize specs with history OR installed pkgs if we are pruning
        # Otherwise, initialization depends on whether we have a history to work with or not
        elif (
            self.solver_input_state.history
            and not self.solver_input_state.update_modifier.UPDATE_ALL
        ):
            # add in historically-requested specs
            self.specs.update(self.solver_input_state.history, reason="As in history")
            for name, record in self.solver_input_state.installed.items():
                if name in self.solver_input_state.aggressive_updates:
                    self.specs.set(
                        name, MatchSpec(name), reason="Installed and in aggressive updates"
                    )
                elif name in self.solver_input_state.do_not_remove:
                    # these are things that we want to keep even if they're not explicitly
                    # specified.  This is to compensate for older installers not recording these
                    # appropriately for them to be preserved.
                    self.specs.set(
                        name,
                        MatchSpec(name),
                        reason="Installed and protected in do_not_remove",
                        overwrite=False,
                    )
                elif record.subdir == "pypi":
                    # add in foreign stuff (e.g. from pip) into the specs
                    # map. We add it so that it can be left alone more. This is a
                    # declaration that it is manually installed, much like the
                    # history map. It may still be replaced if it is in conflict,
                    # but it is not just an indirect dep that can be pruned.
                    self.specs.set(
                        name,
                        MatchSpec(name),
                        reason="Installed from PyPI; protect from indirect pruning",
                    )
        else:
            # add everything in prefix if we have no history to work with (e.g. with --update-all)
            self.specs.update(
                {name: MatchSpec(name) for name in self.solver_input_state.installed},
                reason="Installed and no history available (prune=false)",
            )

        # Add virtual packages so they are taken into account by the solver
        for name in self.solver_input_state.virtual:
            # we only add a bare name spec here, no constrain! the constrain is only available
            # in the index, since it will only contain a single value for the virtual package
            self.specs.set(name, MatchSpec(name), reason="Virtual system", overwrite=False)

    @property
    def current_solution(self):
        """
        Massage currently stored records so they can be returned as the type expected by the
        solver API. This is what you should return in ``Solver.solve_final_state()``.
        """
        return IndexedSet(PrefixGraph(self.records.values()).graph)

    @property
    def real_specs(self):
        """
        Specs that are _not_ virtual.
        """
        return {name: spec for name, spec in self.specs.items() if not name.startswith("__")}

    @property
    def virtual_specs(self):
        """
        Specs that are virtual.
        """
        return {name: spec for name, spec in self.specs.items() if name.startswith("__")}

    def prepare_specs(self, index: IndexHelper) -> Mapping[str, MatchSpec]:
        """
        Main method to populate the ``specs`` mapping.
        """
        if self.solver_input_state.is_removing:
            self._prepare_for_remove()
        else:
            self._prepare_for_add(index)
        self._prepare_for_solve(index)
        return self.specs

    def _prepare_for_add(self, index: IndexHelper):
        """
        This is the core logic for specs processing. In contrast with the ``conda remove``
        logic, this part is more intricate since it has to deal with details such as the
        role of the history specs, aggressive updates, pinned packages and the deps / update
        modifiers.

        Parameters
        ----------
        index
            Needed to query for the dependency tree of potentially conflicting
            specs.
        """
        sis = self.solver_input_state

        # The constructor should have prepared the _basics_ of the specs / records maps. Now we
        # we will try to refine the version constrains to minimize changes in the environment
        # whenever possible. Take into account this is done iteratively together with the
        # solver! self.records starts with the initial prefix state (if any), but accumulates
        # solution attempts after each retry.

        # ## Refine specs that match currently proposed solution
        # ## (either prefix as is, or a failed attempt)

        # First, let's see if the current specs are compatible with the current records. They
        # should be unless something is very wrong with the prefix.

        for name, spec in self.specs.items():
            record_matches = [record for record in self.records.values() if spec.match(record)]

            if not record_matches:
                continue  # nothing to refine

            if len(record_matches) != 1:  # something is very wrong!
                self._raise_incompatible_spec_records(spec, record_matches)

            # ok, now we can start refining
            record = record_matches[0]
            if record.is_unmanageable:
                self.specs.set(
                    name, record.to_match_spec(), reason="Spec matches unmanageable record"
                )
            elif name in sis.aggressive_updates:
                self.specs.set(
                    name, MatchSpec(name), reason="Spec matches record in aggressive updates"
                )
            elif name not in self.conflicts:
                # TODO: and (name not in explicit_pool or record in explicit_pool[name]):
                self.specs.set(
                    name,
                    record.to_match_spec(),
                    reason="Spec matches record in explicit pool for its name",
                )
            elif name in sis.history:
                # if the package was historically requested, we will honor that, but trying to
                # keep the package as installed
                #
                # TODO: JRG: I don't know how mamba will handle _both_ a constrain and a target;
                # play with priorities?
                self.specs.set(
                    name,
                    MatchSpec(sis.history[name], target=record.dist_str()),
                    reason="Spec matches record in history",
                )
            else:
                # every other spec that matches something installed will be configured with
                # only a target. This is the case for conflicts, among others
                self.specs.set(
                    name, MatchSpec(name, target=record.dist_str()), reason="Spec matches record"
                )

        # ## Pinnings ###

        # Now let's add the pinnings
        # We want to pin packages that are
        # - installed
        # - requested by the user (if request and pin conflict, request takes precedence)
        # - a dependency of something requested by the user
        pin_overrides = set()
        # The block using this object below has been deactivated.
        # so we don't build this (potentially expensive) set anymore
        # if sis.pinned:
        #     explicit_pool = set(index.explicit_pool(sis.requested.values()))
        for name, spec in sis.pinned.items():
            pin = MatchSpec(spec, optional=False)
            requested = name in sis.requested
            if name in sis.installed and not requested:
                self.specs.set(name, pin, reason="Pinned, installed and not requested")
            elif requested:
                # THIS BLOCK WOULD NEVER RUN
                # classic solver would check compatibility between pinned and requested
                # and let the user override pins in the CLI. libmamba doesn't allow
                # the user to override pins. We will have raised an exception earlier
                # We will keep this code here for reference
                if True:  # compatible_specs(index, sis.requested[name], spec):
                    # assume compatible, we will raise later otherwise
                    reason = (
                        "Pinned, installed and requested; constraining request "
                        "as pin because they are compatible"
                    )
                    self.specs.set(name, pin, reason=reason)
                    pin_overrides.add(name)
                else:
                    reason = (
                        "Pinned, installed and requested; pin and request "
                        "are conflicting, so adding user request due to higher precedence"
                    )
                    self.specs.set(name, sis.requested[name], reason=reason)
            # always assume the pin will be needed
            # elif name in explicit_pool:
            # THIS BLOCK HAS BEEN DEACTIVATED
            # the explicit pool is potentially expensive and we are not using it.
            # leaving this here for reference. It's supposed to check whether the pin
            # was going to be part of the environment because it shows up in the dependency
            # tree of the explicitly requested specs.
            # ---
            # TODO: This might be introducing additional specs into the list if the pin
            # matches a dependency of a request, but that dependency only appears in _some_
            # of the request variants. For example, package A=2 depends on B, but package
            # A=3 no longer depends on B. B will be part of A's explicit pool because it
            # "could" be a dependency. If B happens to be pinned but A=3 ends up being the
            # one chosen by the solver, then B would be included in the solution when it
            # shouldn't. It's a corner case but it can happen so we might need to further
            # restrict the explicit_pool to see. The original logic in the classic solver
            # checked: `if explicit_pool[s.name] & ssc.r._get_package_pool([s]).get(s.name,
            # set()):`
            else:  # always add the pin for libmamba to consider it
                self.specs.set(
                    name,
                    pin,
                    reason="Pin matches one of the potential dependencies of user requests",
                )
            # In classic, this would notify the pin was being overridden by a request
            # else:
            #     log.warn(
            #         "pinned spec %s conflicts with explicit specs. Overriding pinned spec.", spec
            #     )

        # ## Update modifiers ###

        if sis.update_modifier.FREEZE_INSTALLED:
            for name, record in sis.installed.items():
                if name in self.conflicts:
                    # TODO: Investigate why we use to_match_spec() here and other targets use
                    # dist_str()
                    self.specs.set(
                        name,
                        MatchSpec(name, target=record.to_match_spec(), optional=True),
                        reason="Relaxing installed because it caused a conflict",
                    )
                else:
                    self.specs.set(name, record.to_match_spec(), reason="Freezing as installed")

        elif sis.update_modifier.UPDATE_ALL:
            # NOTE: This logic is VERY similar to what we are doing in the class constructor (?)
            # NOTE: we are REDEFINING the specs accumulated so far
            old_specs = self.specs._data.copy()
            self.specs.clear(reason="Redefining from scratch due to --update-all")
            if sis.history:
                # history is preferable because it has explicitly installed stuff in it.
                # that simplifies our solution.
                for name in sis.history:
                    if name in sis.pinned:
                        self.specs.set(
                            name,
                            old_specs[name],
                            reason="Update all, with history, pinned: reusing existing entry",
                        )
                    else:
                        self.specs.set(
                            name,
                            MatchSpec(name),
                            reason="Update all, with history, not pinned: adding spec "
                            "from history with no constraints",
                        )

                for name, record in sis.installed.items():
                    if record.subdir == "pypi":
                        self.specs.set(
                            name,
                            MatchSpec(name),
                            reason="Update all, with history: treat pip installed "
                            "stuff as explicitly installed",
                        )
                    elif name not in self.specs:
                        self.specs.set(
                            name,
                            MatchSpec(name),
                            reason="Update all, with history: "
                            "adding name-only spec from installed",
                        )
            else:
                for name in sis.installed:
                    if name in sis.pinned:
                        self.specs.set(
                            name,
                            old_specs[name],
                            reason="Update all, no history, pinned: reusing existing entry",
                        )
                    else:
                        self.specs.set(
                            name,
                            MatchSpec(name),
                            reason="Update all, no history, not pinned: adding spec from "
                            "installed with no constraints",
                        )

        elif sis.update_modifier.UPDATE_SPECS:
            # this is the default behavior if no flags are passed
            # NOTE: This _anticipates_ conflicts; we can also wait for the next attempt and
            # get the real solver conflicts as part of self.conflicts -- that would simplify
            # this logic a bit

            # ensure that our self.specs_to_add are not being held back by packages in the env.
            # This factors in pins and also ignores specs from the history.  It is unfreezing
            # only for the indirect specs that otherwise conflict with update of the immediate
            # request:
            # pinned_requests = []
            # for name, spec in sis.requested.items():
            #     if name not in pin_overrides and name in sis.pinned:
            #         continue
            #     if name in sis.history:
            #         continue
            #     pinned_requests.append(sis.package_has_updates(spec))
            #     # this ^ needs to be implemented, requires installed pool
            # conflicts = sis.get_conflicting_specs(self.specs.values(), pinned_requests) or ()
            for name in self.conflicts:
                if (
                    name not in sis.pinned
                    and name not in sis.history
                    and name not in sis.requested
                ):
                    self.specs.set(name, MatchSpec(name), reason="Relaxed because conflicting")

        # ## Python pinning ###

        # As a business rule, we never want to update python beyond the current minor version,
        # unless that's requested explicitly by the user (which we actively discourage).

        if "python" in self.records and "python" not in sis.requested:
            record = self.records["python"]
            if "python" not in self.conflicts and sis.update_modifier.FREEZE_INSTALLED:
                self.specs.set(
                    "python",
                    record.to_match_spec(),
                    reason="Freezing python due to business rule, freeze-installed, "
                    "and no conflicts",
                )
            else:
                # will our prefix record conflict with any explicit spec?  If so, don't add
                # anything here - let python float when it hasn't been explicitly specified
                spec = self.specs.get("python", MatchSpec("python"))
                if spec.get("version"):
                    reason = "Leaving Python pinning as it was calculated so far"
                else:
                    reason = "Pinning Python to match installed version"
                    version = get_major_minor_version(record.version) + ".*"
                    spec = MatchSpec(spec, version=version)

                # There's a chance the selected version results in a conflict -- detect and
                # report?
                # specs = (spec, ) + tuple(sis.requested.values())
                # if sis.get_conflicting_specs(specs, sis.requested.values()):
                #     if not sis.installing:  # TODO: repodata checks?
                #         # raises a hopefully helpful error message
                #         sis.find_conflicts(specs)  # this might call the solver -- remove?
                #     else:
                #         # oops, no message?
                #         raise LibMambaUnsatisfiableError(
                #             "Couldn't find a Python version that does not conflict..."
                #         )

                self.specs.set("python", spec, reason=reason)

        # ## Offline and aggressive updates ###

        # For the aggressive_update_packages configuration parameter, we strip any target
        # that's been set.

        if not context.offline:
            for name, spec in sis.aggressive_updates.items():
                if name in self.specs:
                    self.specs.set(name, spec, reason="Aggressive updates relaxation")

        # ## User requested specs ###

        # add in explicitly requested specs from specs_to_add
        # this overrides any name-matching spec already in the spec map

        for name, spec in sis.requested.items():
            if name not in pin_overrides:
                self.specs.set(name, spec, reason="Explicitly requested by user")

        # ## Conda pinning ###

        # As a business rule, we never want to downgrade conda below the current version,
        # unless that's requested explicitly by the user (which we actively discourage).

        if (
            "conda" in self.specs
            and "conda" in sis.installed
            and paths_equal(sis.prefix, context.conda_prefix)
        ):
            record = sis.installed["conda"]
            spec = self.specs["conda"]
            required_version = f">={record.version}"
            if not spec.get("version"):
                spec = MatchSpec(spec, version=required_version)
                reason = "Pinning conda with version greater than currently installed"
                self.specs.set("conda", spec, reason=reason)
            if context.auto_update_conda and "conda" not in sis.requested:
                spec = MatchSpec("conda", version=required_version, target=None)
                reason = "Pinning conda with version greater than currently installed, auto update"
                self.specs.set("conda", spec, reason=reason)

        # ## Extra logic ###
        # this is where we are adding workarounds for mamba difference's in behavior, which
        # might not belong here as they are solver specific

        # next step -> .prepare_for_solve()

    def _prepare_for_remove(self):
        "Just add the user requested specs to ``specs``"
        # This logic is simpler than when we are installing packages
        self.specs.update(self.solver_input_state.requested, reason="Adding user-requested specs")

    def _prepare_for_solve(self, index):
        """
        Last part of the logic, common to addition and removal of packages. Originally,
        the legacy logic will also minimize the conflicts here by doing a pre-solve
        analysis, but so far we have opted for a different approach in libmamba: let the
        solver try, fail with conflicts, and annotate those as such so they are unconstrained.

        Now, this method only ensures that the pins do not cause conflicts.
        """
        # ## Inconsistency analysis ###
        # here we would call conda.core.solve.classic.Solver._find_inconsistent_packages()

        # ## Check pin and requested are compatible
        sis = self.solver_input_state
        requested_and_pinned = set(sis.requested).intersection(sis.pinned)
        for name in requested_and_pinned:
            requested = sis.requested[name]
            pin = sis.pinned[name]
            installed = sis.installed.get(name)
            if (
                # name-only pins lock to installed; requested spec must match it
                (pin.is_name_only_spec and installed and not requested.match(installed))
                # otherwise, the pin needs to be compatible with the requested spec
                or not compatible_specs(index, (requested, pin))
            ):
                pinned_specs = [
                    (sis.installed.get(name, pin) if pin.is_name_only_spec else pin)
                    for name, pin in sorted(sis.pinned.items())
                ]
                exc = SpecsConfigurationConflictError(
                    requested_specs=sorted(sis.requested.values(), key=lambda x: x.name),
                    pinned_specs=pinned_specs,
                    prefix=sis.prefix,
                )
                exc.allow_retry = False
                raise exc

        # ## Conflict minimization ###
        # here conda.core.solve.classic.Solver._run_sat() enters a `while conflicting_specs` loop
        # to neuter some of the specs in self.specs. In other solvers we let the solver run into
        # them. We might need to add a hook here ?

        # After this, we finally let the solver do its work. It will either finish with a final
        # state or fail and repopulate the conflicts list in the SolverOutputState object

    def early_exit(self):
        """
        Operations that do not need a solver and might result in returning
        early are collected here.
        """
        sis = self.solver_input_state
        if sis.is_removing:
            not_installed = [
                spec for name, spec in sis.requested.items() if name not in sis.installed
            ]
            if not_installed:
                exc = PackagesNotFoundError(not_installed)
                exc.allow_retry = False
                raise exc

            if sis.force_remove:
                for name, spec in sis.requested.items():
                    for record in sis.installed.values():
                        if spec.match(record):
                            self.records.pop(name)
                            break
                return self.current_solution

        if sis.update_modifier.SPECS_SATISFIED_SKIP_SOLVE and not sis.is_removing:
            for name, spec in sis.requested.items():
                if name not in sis.installed:
                    break
            else:
                # All specs match a package in the current environment.
                # Return early, with the current solution (at this point, .records is set
                # to the map of installed packages)
                return self.current_solution

    def post_solve(self, solver: Type["Solver"]):
        """
        These tasks are performed _after_ the solver has done its work. It essentially
        post-processes the ``records`` mapping.

        Parameters
        ----------
        solver_cls
            The class used to instantiate the Solver. If not provided, defaults to the one
            specified in the context configuration.

        Notes
        -----
        This method could be solver-agnostic  but unfortunately ``--update-deps`` requires a
        second solve; that's why this method needs a solver class to be passed as an argument.
        """
        # After a solve, we still need to do some refinement
        sis = self.solver_input_state

        # ## Record history ###
        # user requested specs need to be annotated in history
        # we control that in .for_history
        self.for_history.update(sis.requested, reason="User requested specs recorded to history")

        # ## Neutered ###
        # annotate overridden history specs so they are written to disk
        for name, spec in sis.history.items():
            record = self.records.get(name)
            if record and not spec.match(record):
                self.neutered.set(
                    name,
                    MatchSpec(name, version=record.version),
                    reason="Solution required a history override",
                )

        # ## Add inconsistent packages back ###
        # direct result of the inconsistency analysis above

        # ## Deps modifier ###
        # handle the different modifiers (NO_DEPS, ONLY_DEPS, UPDATE_DEPS)
        # this might mean removing different records by hand or even calling
        # the solver a 2nd time

        if sis.deps_modifier.NO_DEPS:
            # In the NO_DEPS case, we need to start with the original list of packages in the
            # environment, and then only modify packages that match the requested specs
            #
            # Help information notes that use of NO_DEPS is expected to lead to broken
            # environments.
            original_state = dict(sis.installed)
            only_change_these = {}
            for name, spec in sis.requested.items():
                for record in self.records.values():
                    if spec.match(record):
                        only_change_these[name] = record

            if sis.is_removing:
                # TODO: This could be a pre-solve task to save time in forced removes?
                for name in only_change_these:
                    del original_state[name]
            else:
                for name, record in only_change_these.items():
                    original_state[name] = record

            self.records.clear(reason="Redefining records due to --no-deps")
            self.records.update(original_state, reason="Redefined records due to --no-deps")

        elif sis.deps_modifier.ONLY_DEPS and not sis.update_modifier.UPDATE_DEPS:
            # Using a special instance of PrefixGraph to remove youngest child nodes that match
            # the original requested specs.  It's important to remove only the *youngest* child
            # nodes, because a typical use might be `conda install --only-deps python=2 flask`,
            # and in that case we'd want to keep python.
            #
            # What are we supposed to do if flask was already in the environment?
            # We can't be removing stuff here that's already in the environment.
            #
            # What should be recorded for the user-requested specs in this case? Probably all
            # direct dependencies of flask.

            graph = PrefixGraph(self.records.values(), sis.requested.values())
            # this method below modifies the graph inplace _and_ returns the removed nodes
            # (like dict.pop())
            would_remove = graph.remove_youngest_descendant_nodes_with_specs()

            # We need to distinguish the behavior between `conda remove` and the rest
            to_remove = []
            if sis.is_removing:
                for record in would_remove:
                    # do not remove records that were not requested but were installed
                    if record.name not in sis.requested and record.name in sis.installed:
                        continue
                    to_remove.append(record.name)
            else:
                for record in would_remove:
                    for dependency in record.depends:
                        spec = MatchSpec(dependency)
                        if spec.name not in self.specs:
                            # following https://github.com/conda/conda/pull/8766
                            reason = "Recording deps brought by --only-deps as explicit"
                            self.for_history.set(spec.name, spec, reason=reason)
                    to_remove.append(record.name)

            for name in to_remove:
                installed = sis.installed.get(name)
                if installed:
                    self.records.set(
                        name, installed, reason="Restoring originally installed due to --only-deps"
                    )
                else:
                    self.records.pop(
                        record.name, reason="Excluding from solution due to --only-deps"
                    )

        elif sis.update_modifier.UPDATE_DEPS:
            # Here we have to SAT solve again :(  It's only now that we know the dependency
            # chain of specs_to_add.
            #
            # UPDATE_DEPS is effectively making each spec in the dependency chain a
            # user-requested spec. For all other specs, we drop all information but name, drop
            # target, and add them to `requested` so it gets recorded in the history file.
            #
            # It's like UPDATE_ALL, but only for certain dependency chains.
            new_specs = TrackedMap("update_deps_specs")

            graph = PrefixGraph(self.records.values())
            for name, spec in sis.requested.items():
                record = graph.get_node_by_name(name)
                for ancestor in graph.all_ancestors(record):
                    new_specs.set(
                        ancestor.name,
                        MatchSpec(ancestor.name),
                        reason="New specs asked by --update-deps",
                    )

            # Remove pinned_specs
            for name, spec in sis.pinned.items():
                new_specs.pop(
                    name, None, reason="Exclude pinned packages from --update-deps specs"
                )
            # Follow major-minor pinning business rule for python
            if "python" in new_specs:
                record = sis.installed["python"]
                version = ".".join(record.version.split(".")[:2]) + ".*"
                new_specs.set("python", MatchSpec(name="python", version=version))
            # Add in the original `requested` on top.
            new_specs.update(
                sis.requested, reason="Add original requested specs on top for --update-deps"
            )

            if sis.is_removing:
                specs_to_add = ()
                specs_to_remove = list(new_specs.keys())
            else:
                specs_to_add = list(new_specs.values())
                specs_to_remove = ()

            with context._override("quiet", False):
                # Create a new solver instance to perform a 2nd solve with deps added We do it
                # like this to avoid overwriting state accidentally. Instead, we will import
                # the needed state bits manually.
                records = solver.__class__(
                    prefix=solver.prefix,
                    channels=solver.channels,
                    subdirs=solver.subdirs,
                    specs_to_add=specs_to_add,
                    specs_to_remove=specs_to_remove,
                    command="recursive_call_for_update_deps",
                ).solve_final_state(
                    update_modifier=UpdateModifier.UPDATE_SPECS,  # avoid recursion!
                    deps_modifier=sis._deps_modifier,
                    ignore_pinned=sis.ignore_pinned,
                    force_remove=sis.force_remove,
                    prune=sis.prune,
                )
                records = {record.name: record for record in records}

            self.records.clear(reason="Redefining due to --update-deps")
            self.records.update(records, reason="Redefined due to --update-deps")
            self.for_history.clear(reason="Redefining due to --update-deps")
            self.for_history.update(new_specs, reason="Redefined due to --update-deps")

            # Disable pruning regardless the original value
            # TODO: Why? Dive in https://github.com/conda/conda/pull/7719
            sis._prune = False

        # ## Prune ###
        # remove orphan leaves in the graph
        if sis.prune:
            graph = PrefixGraph(list(self.records.values()), self.specs.values())
            graph.prune()
            self.records.clear(reason="Pruning")
            self.records.update({record.name: record for record in graph.graph}, reason="Pruned")

    @staticmethod
    def _raise_incompatible_spec_records(spec, records):
        "Raise an error if something is very wrong with the environment"
        raise CondaError(
            dals(
                f"""
                Conda encountered an error with your environment.  Please report an issue
                at https://github.com/conda/conda/issues.  In your report, please include
                the output of 'conda info' and 'conda list' for the active environment, along
                with the command you invoked that resulted in this error.
                pkg_name: {spec.name}
                spec: {spec}
                matches_for_spec: {dashlist(records, indent=4)}
                """
            )
        )
