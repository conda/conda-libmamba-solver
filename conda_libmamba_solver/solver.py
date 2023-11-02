# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
This module defines the conda.core.solve.Solver interface and its immediate helpers

We can import from conda and libmambapy. `mamba` itself should NOT be imported here.
"""
import json
import logging
import os
import re
import sys
from collections import defaultdict
from contextlib import suppress
from functools import lru_cache
from inspect import stack
from textwrap import dedent
from typing import Iterable, Mapping, Optional, Sequence, Union

import libmambapy as api
from boltons.setutils import IndexedSet
from conda import __version__ as _conda_version
from conda.base.constants import (
    DEFAULT_CHANNELS,
    REPODATA_FN,
    UNKNOWN_CHANNEL,
    ChannelPriority,
    on_win,
)
from conda.base.context import context
from conda.common.constants import NULL
from conda.common.io import Spinner, timeout
from conda.common.path import paths_equal
from conda.common.url import join_url, percent_decode
from conda.core.prefix_data import PrefixData
from conda.core.solve import Solver
from conda.exceptions import (
    InvalidMatchSpec,
    PackagesNotFoundError,
    SpecsConfigurationConflictError,
    UnsatisfiableError,
)
from conda.models.channel import Channel
from conda.models.match_spec import MatchSpec
from conda.models.records import PackageRecord, PrefixRecord
from conda.models.version import VersionOrder

from . import __version__
from .exceptions import LibMambaUnsatisfiableError
from .index import LibMambaIndexHelper, _CachedLibMambaIndexHelper
from .mamba_utils import init_api_context, mamba_version
from .state import SolverInputState, SolverOutputState
from .utils import is_channel_available

log = logging.getLogger(f"conda.{__name__}")


class LibMambaSolver(Solver):
    """
    Cleaner implementation using the ``state`` module helpers.
    """

    _uses_ssc = False

    def __init__(
        self,
        prefix,
        channels,
        subdirs=(),
        specs_to_add=(),
        specs_to_remove=(),
        repodata_fn=REPODATA_FN,
        command=NULL,
    ):
        if specs_to_add and specs_to_remove:
            raise ValueError(
                "Only one of `specs_to_add` and `specs_to_remove` can be set at a time"
            )
        if specs_to_remove and command is NULL:
            command = "remove"

        super().__init__(
            prefix,
            channels,
            subdirs=subdirs,
            specs_to_add=specs_to_add,
            specs_to_remove=specs_to_remove,
            repodata_fn=repodata_fn,
            command=command,
        )

        if self.subdirs is NULL or not self.subdirs:
            self.subdirs = context.subdirs

        # These three attributes are set during ._setup_solver()
        self.solver = None
        self._solver_options = None

        # we want to support arbitrary repodata fns, but we ignore current_repodata
        if self._repodata_fn == "current_repodata.json":
            log.debug(f"Ignoring repodata_fn='current_repodata.json', defaulting to {REPODATA_FN}")
            self._repodata_fn = REPODATA_FN

        # Fix bug in conda.common.arg2spec and MatchSpec.__str__
        fixed_specs = []
        for spec in specs_to_add:
            if isinstance(spec, PackageRecord):
                spec = MatchSpec(str(spec))
            elif isinstance(spec, MatchSpec):
                spec_str = str(spec)
                if "::" in spec_str:
                    for arg in sys.argv:
                        if spec_str in arg:
                            ms_from_arg = MatchSpec(arg)
                            if ms_from_arg.name == spec.name:
                                spec = ms_from_arg
            fixed_specs.append(spec)
        # MatchSpec.merge sorts before merging; keep order without dups with IndexedSet
        self.specs_to_add = IndexedSet(MatchSpec.merge(s for s in fixed_specs))

    @staticmethod
    @lru_cache(maxsize=None)
    def user_agent():
        """
        Expose this identifier to allow conda to extend its user agent if required
        """
        return f"conda-libmamba-solver/{__version__} libmambapy/{mamba_version()}"

    def solve_final_state(
        self,
        update_modifier=NULL,
        deps_modifier=NULL,
        prune=NULL,
        ignore_pinned=NULL,
        force_remove=NULL,
        should_retry_solve=False,
    ):
        self._log_info()
        in_state = SolverInputState(
            prefix=self.prefix,
            requested=self.specs_to_add or self.specs_to_remove,
            update_modifier=update_modifier,
            deps_modifier=deps_modifier,
            prune=prune,
            ignore_pinned=ignore_pinned,
            force_remove=force_remove,
            command=self._command,
        )

        out_state = SolverOutputState(solver_input_state=in_state)

        # These tasks do _not_ require a solver...
        # TODO: Abstract away in the base class?
        none_or_final_state = out_state.early_exit()
        if none_or_final_state is not None:
            return none_or_final_state

        # From now on we _do_ require a solver and the index
        init_api_context()
        subdirs = self.subdirs
        conda_bld_channels = ()
        if self._called_from_conda_build():
            log.info("Using solver via 'conda.plan.install_actions' (probably conda build)")
            # Problem: Conda build generates a custom index which happens to "forget" about
            # noarch on purpose when creating the build/host environments, since it merges
            # both as if they were all in the native subdir. this causes package-not-found
            # errors because we are not using the patched index.
            # Fix: just add noarch to subdirs.
            if "noarch" not in subdirs:
                subdirs = *subdirs, "noarch"
            # We need to recover the local dirs (conda-build's local, output_folder, etc)
            # from the index. This is a bit of a hack, but it works.
            conda_bld_channels = {
                rec.channel: None for rec in self._index if rec.channel.scheme == "file"
            }
            # Cache indices for conda-build, it gets heavy otherwise
            IndexHelper = _CachedLibMambaIndexHelper
        else:
            IndexHelper = LibMambaIndexHelper

        all_channels = [
            *conda_bld_channels,
            *self.channels,
            *in_state.channels_from_specs(),
        ]
        override = (getattr(context, "_argparse_args", None) or {}).get("override_channels")
        if not os.getenv("CONDA_LIBMAMBA_SOLVER_NO_CHANNELS_FROM_INSTALLED") and not override:
            # see https://github.com/conda/conda-libmamba-solver/issues/108
            all_urls = [url for c in all_channels for url in Channel(c).urls(False)]
            installed_channels = in_state.channels_from_installed(seen=all_urls)
            for channel in installed_channels:
                # Only add to list if resource is available; check has timeout=1s
                if timeout(1, is_channel_available, channel.base_url, default_return=False):
                    all_channels.append(channel)
        all_channels.extend(in_state.maybe_free_channel())

        all_channels = tuple(all_channels)
        with Spinner(
            self._spinner_msg_metadata(all_channels, conda_bld_channels=conda_bld_channels),
            enabled=not context.verbosity and not context.quiet,
            json=context.json,
        ):
            index = IndexHelper(
                installed_records=(*in_state.installed.values(), *in_state.virtual.values()),
                channels=all_channels,
                subdirs=subdirs,
                repodata_fn=self._repodata_fn,
            )
            index.reload_local_channels()

        with Spinner(
            self._spinner_msg_solving(),
            enabled=not context.verbosity and not context.quiet,
            json=context.json,
        ):
            self._setup_solver(index)
            # This function will copy and mutate `out_state`
            # Make sure we get the latest copy to return the correct solution below
            out_state = self._solving_loop(in_state, out_state, index)
            self.neutered_specs = tuple(out_state.neutered.values())
            solution = out_state.current_solution

        # Check whether conda can be updated; this is normally done in .solve_for_diff()
        # but we are doing it now so we can reuse in_state and friends
        self._notify_conda_outdated(None, index, solution)

        return solution

    def _spinner_msg_metadata(self, channels: Iterable[Channel], conda_bld_channels=()):
        if self._called_from_conda_build():
            msg = "Reloading output folder"
            if conda_bld_channels:
                names = [Channel(c).canonical_name for c in conda_bld_channels]
                msg += f" ({', '.join(names)})"
            return msg
        canonical_names = list(dict.fromkeys([c.canonical_name for c in channels]))
        canonical_names_dashed = "\n - ".join(canonical_names)
        return (
            f"Channels:\n"
            f" - {canonical_names_dashed}\n"
            f"Platform: {context.subdir}\n"
            f"Collecting package metadata ({self._repodata_fn})"
        )

    def _spinner_msg_solving(self):
        """This shouldn't be our responsibility, but the CLI / app's..."""
        prefix_name = os.path.basename(self.prefix)
        if self._called_from_conda_build():
            if "_env" in prefix_name:
                env_name = "_".join(prefix_name.split("_")[:3])
                return f"Solving environment ({env_name})"
            else:
                # https://github.com/conda/conda-build/blob/e0884b626a/conda_build/environ.py#L1035-L1036
                return "Getting pinned dependencies"
        return "Solving environment"

    def _solving_loop(
        self,
        in_state: SolverInputState,
        out_state: SolverOutputState,
        index: LibMambaIndexHelper,
    ):
        solved = False
        max_attempts = max(
            2,
            int(os.environ.get("CONDA_LIBMAMBA_SOLVER_MAX_ATTEMPTS", len(in_state.installed))) + 1,
        )
        for attempt in range(1, max_attempts):
            log.debug("Starting solver attempt %s", attempt)
            try:
                solved = self._solve_attempt(in_state, out_state, index)
                if solved:
                    break
            except (UnsatisfiableError, PackagesNotFoundError):
                solved = False
                break  # try with last attempt
            else:  # didn't solve yet, but can retry
                out_state = SolverOutputState(
                    solver_input_state=in_state,
                    specs=dict(out_state.specs),
                    records=dict(out_state.records),
                    for_history=dict(out_state.for_history),
                    neutered=dict(out_state.neutered),
                    conflicts=dict(out_state.conflicts),
                )
        if not solved:
            log.debug("Last attempt: reporting all installed as conflicts")
            out_state.conflicts.update(
                {
                    name: record.to_match_spec()
                    for name, record in in_state.installed.items()
                    # TODO: These conditions might not be needed here
                    if not record.is_unmanageable
                    # or name not in in_state.history
                    # or name not in in_state.requested
                    # or name not in in_state.pinned
                },
                reason="Last attempt: all installed packages exposed "
                "as conflicts for maximum flexibility",
            )
            # we only check this for "desperate" strategies in _specs_to_tasks
            if self._command in (None, NULL):
                self._command = "last_solve_attempt"
            else:
                self._command += "+last_solve_attempt"
            solved = self._solve_attempt(in_state, out_state, index)
            if not solved:
                message = self._prepare_problems_message(pins=out_state.pins)
                exc = LibMambaUnsatisfiableError(message)
                exc.allow_retry = False
                raise exc

        # We didn't fail? Nice, let's return the calculated state
        self._export_solved_records(in_state, out_state, index)

        # Run post-solve tasks
        out_state.post_solve(solver=self)

        return out_state

    def _log_info(self):
        log.info("Using libmamba solver")
        log.info("Conda version: %s", _conda_version)
        log.info("Mamba version: %s", mamba_version())
        log.info("Target prefix: %r", self.prefix)
        log.info("Command: %s", sys.argv)
        log.info("Specs to add: %s", self.specs_to_add)
        log.info("Specs to remove: %s", self.specs_to_remove)

    def _setup_solver(self, index: LibMambaIndexHelper):
        self._solver_options = solver_options = [
            (api.SOLVER_FLAG_ALLOW_DOWNGRADE, 1),
        ]
        if context.channel_priority is ChannelPriority.STRICT:
            solver_options.append((api.SOLVER_FLAG_STRICT_REPO_PRIORITY, 1))
        if self.specs_to_remove and self._command in ("remove", None, NULL):
            solver_options.append((api.SOLVER_FLAG_ALLOW_UNINSTALL, 1))

        self.solver = api.Solver(index._pool, self._solver_options)

    def _solve_attempt(
        self,
        in_state: SolverInputState,
        out_state: SolverOutputState,
        index: LibMambaIndexHelper,
    ):
        self._setup_solver(index)

        log.debug("New solver attempt")
        log.debug("Current conflicts (including learnt ones): %s", out_state.conflicts)

        # ## First, we need to obtain the list of specs ###
        try:
            out_state.prepare_specs(index)
        except SpecsConfigurationConflictError as exc:
            # in the last attempt we have marked everything
            # as a conflict so everything gets unconstrained
            # however this will be detected as a conflict with the
            # pins, but we can ignore it because we did it ourselves
            if self._command != "last_solve_attempt":
                raise exc

        log.debug("Computed specs: %s", out_state.specs)

        # ## Convert to tasks
        out_state.pins.clear()
        n_pins = 0
        tasks = self._specs_to_tasks(in_state, out_state)
        for (task_name, task_type), specs in tasks.items():
            log.debug("Adding task %s with specs %s", task_name, specs)
            if task_name == "ADD_PIN":
                for spec in specs:
                    n_pins += 1
                    self.solver.add_pin(spec)
                    out_state.pins[f"pin-{n_pins}"] = spec
            else:
                self.solver.add_jobs(specs, task_type)

        # ## Run solver
        solved = self.solver.solve()

        if solved:
            out_state.conflicts.clear(reason="Solution found")
            return solved

        problems = self.solver.problems_to_str()
        old_conflicts = out_state.conflicts.copy()
        new_conflicts = self._maybe_raise_for_problems(
            problems, old_conflicts, out_state.pins, index._channels
        )
        log.debug("Attempt failed with %s conflicts", len(new_conflicts))
        out_state.conflicts.update(new_conflicts.items(), reason="New conflict found")
        return False

    def _specs_to_tasks(self, in_state: SolverInputState, out_state: SolverOutputState):
        log.debug("Creating tasks for %s specs", len(out_state.specs))
        if in_state.is_removing:
            return self._specs_to_tasks_remove(in_state, out_state)
        if self._called_from_conda_build():
            return self._specs_to_tasks_conda_build(in_state, out_state)
        return self._specs_to_tasks_add(in_state, out_state)

    @staticmethod
    def _spec_to_str(spec):
        """
        Workarounds for Matchspec str-roundtrip limitations.

        Note: this might still fail for specs with local channels and version=*:
            file://path/to/channel::package_name=*=*buildstr*
        """
        if spec.original_spec_str and spec.original_spec_str.startswith("file://"):
            return spec.original_spec_str
        if spec.get("build") and not spec.get("version"):
            spec = MatchSpec(spec, version="*")
        return str(spec)

    def _specs_to_tasks_add(self, in_state: SolverInputState, out_state: SolverOutputState):
        tasks = defaultdict(list)

        # Protect history and aggressive updates from being uninstalled if possible. From libsolv
        # docs: "The matching installed packages are considered to be installed by a user, thus not
        # installed to fulfill some dependency. This is needed input for the calculation of
        # unneeded packages for jobs that have the SOLVER_CLEANDEPS flag set."
        user_installed = {
            pkg
            for pkg in (
                *in_state.history,
                *in_state.aggressive_updates,
                *in_state.pinned,
                *in_state.do_not_remove,
            )
            if pkg in in_state.installed
        }

        # Fast-track python version changes (Part 1/2)
        # ## When the Python version changes, this implies all packages depending on
        # ## python will be reinstalled too. This can mean that we'll have to try for every
        # ## installed package to result in a conflict before we get to actually solve everything
        # ## A workaround is to let all non-noarch python-depending specs to "float" by marking
        # ## them as a conflict preemptively
        python_version_might_change = False
        installed_python = in_state.installed.get("python")
        to_be_installed_python = out_state.specs.get("python")
        if installed_python and to_be_installed_python:
            python_version_might_change = not to_be_installed_python.match(installed_python)

        # Add specs to install
        # SolverOutputState.specs has been populated at initialization with what the classic
        # logic considers should be the target version for each package in the environment
        # and requested changes. We are _not_ following those targets here, but we do iterate
        # over the list to decide what to do with each package.
        for name, _classic_logic_spec in out_state.specs.items():
            if name.startswith("__"):
                continue  # ignore virtual packages
            installed: PackageRecord = in_state.installed.get(name)
            if installed:
                installed_spec_str = self._spec_to_str(installed.to_match_spec())
            else:
                installed_spec_str = None
            requested: MatchSpec = self._check_spec_compat(in_state.requested.get(name))
            history: MatchSpec = self._check_spec_compat(in_state.history.get(name))
            pinned: MatchSpec = self._check_spec_compat(in_state.pinned.get(name))
            conflicting: MatchSpec = self._check_spec_compat(out_state.conflicts.get(name))

            if name in user_installed and not in_state.prune and not conflicting:
                tasks[("USERINSTALLED", api.SOLVER_USERINSTALLED)].append(installed_spec_str)

            # These specs are explicit in some sort of way
            if pinned and not pinned.is_name_only_spec:
                # these are the EXPLICIT pins; conda also uses implicit pinning to
                # constrain updates too but those can be overridden in case of conflicts.
                # name-only pins are treated as locks when installed, see below
                tasks[("ADD_PIN", api.SOLVER_NOOP)].append(self._spec_to_str(pinned))
            # in libmamba, pins and installs are compatible tasks (pin only constrains,
            # does not 'request' a package). In classic, pins were actually targeted installs
            # so they were exclusive
            if requested:
                if requested.is_name_only_spec and pinned and not pinned.is_name_only_spec:
                    # for name-only specs, this is a no-op; we already added the pin above
                    # but we will constrain it again in the install task to have better
                    # error messages if not solvable
                    spec_str = self._spec_to_str(pinned)
                else:
                    spec_str = self._spec_to_str(requested)
                if installed:
                    tasks[("UPDATE", api.SOLVER_UPDATE)].append(spec_str)
                    tasks[("ALLOW_UNINSTALL", api.SOLVER_ALLOWUNINSTALL)].append(name)
                else:
                    tasks[("INSTALL", api.SOLVER_INSTALL)].append(spec_str)
            elif name in in_state.always_update:
                tasks[("UPDATE", api.SOLVER_UPDATE)].append(name)
                tasks[("ALLOW_UNINSTALL", api.SOLVER_ALLOWUNINSTALL)].append(name)
            # These specs are "implicit"; the solver logic massages them for better UX
            # as long as they don't cause trouble
            elif in_state.prune:
                continue
            elif name == "python" and installed and not pinned:
                pyver = ".".join(installed.version.split(".")[:2])
                tasks[("ADD_PIN", api.SOLVER_NOOP)].append(f"python {pyver}.*")
            elif history:
                if conflicting and history.strictness == 3:
                    # relax name-version-build (strictness=3) history specs that cause conflicts
                    # this is called neutering and makes test_neutering_of_historic_specs pass
                    spec = f"{name} {history.version}.*" if history.version else name
                    tasks[("INSTALL", api.SOLVER_INSTALL)].append(spec)
                else:
                    tasks[("INSTALL", api.SOLVER_INSTALL)].append(self._spec_to_str(history))
            elif installed:
                if conflicting:
                    tasks[("ALLOW_UNINSTALL", api.SOLVER_ALLOWUNINSTALL)].append(name)
                else:
                    # we freeze everything else as installed
                    lock = in_state.update_modifier.FREEZE_INSTALLED
                    if pinned and pinned.is_name_only_spec:
                        # name-only pins are treated as locks when installed
                        lock = True
                    if python_version_might_change and installed.noarch is None:
                        for dep in installed.depends:
                            if MatchSpec(dep).name in ("python", "python_abi"):
                                lock = False
                                break
                    if lock:
                        tasks[("LOCK", api.SOLVER_LOCK | api.SOLVER_WEAK)].append(
                            installed_spec_str
                        )

        return dict(tasks)

    def _specs_to_tasks_remove(self, in_state: SolverInputState, out_state: SolverOutputState):
        # TODO: Consider merging add/remove in a single logic this so there's no split

        tasks = defaultdict(list)

        # Protect history and aggressive updates from being uninstalled if possible

        for name, record in out_state.records.items():
            if name in in_state.history or name in in_state.aggressive_updates:
                # MatchSpecs constructed from PackageRecords get parsed too
                # strictly if exported via str(). Use .conda_build_form() directly.
                spec = record.to_match_spec().conda_build_form()
                tasks[("USERINSTALLED", api.SOLVER_USERINSTALLED)].append(spec)

        # No complications here: delete requested and their deps
        # TODO: There are some flags to take care of here, namely:
        # --all
        # --no-deps
        # --deps-only
        key = ("ERASE | CLEANDEPS", api.SOLVER_ERASE | api.SOLVER_CLEANDEPS)
        for name, spec in in_state.requested.items():
            spec = self._check_spec_compat(spec)
            tasks[key].append(str(spec))

        return dict(tasks)

    def _specs_to_tasks_conda_build(
        self, in_state: SolverInputState, out_state: SolverOutputState
    ):
        tasks = defaultdict(list)
        key = "INSTALL", api.SOLVER_INSTALL
        for name, spec in out_state.specs.items():
            if name.startswith("__"):
                continue
            spec = self._check_spec_compat(spec)
            spec = self._fix_version_field_for_conda_build(spec)
            tasks[key].append(spec.conda_build_form())

        return dict(tasks)

    @staticmethod
    def _fix_version_field_for_conda_build(spec: MatchSpec):
        """Fix taken from mambabuild"""
        if spec.version:
            only_dot_or_digit_re = re.compile(r"^[\d\.]+$")
            version_str = str(spec.version)
            if re.match(only_dot_or_digit_re, version_str):
                spec_fields = spec.conda_build_form().split()
                if version_str.count(".") <= 1:
                    spec_fields[1] = version_str + ".*"
                else:
                    spec_fields[1] = version_str + "*"
                return MatchSpec(" ".join(spec_fields))
        return spec

    @staticmethod
    def _str_to_matchspec(spec: Union[str, Sequence[str]]):
        try:
            if isinstance(spec, str):
                name, version, build = spec.rsplit("-", 2)
                return MatchSpec(name=name, version=version, build=build)
            else:
                kwargs = {"name": spec[0].rstrip(",")}
                if len(spec) >= 2:
                    kwargs["version"] = spec[1].rstrip(",")
                if len(spec) == 3:
                    kwargs["build"] = spec[2].rstrip(",")
                return MatchSpec(**kwargs)
        except Exception as exc:
            raise ValueError(f"Could not parse spec: {spec}") from exc

    @classmethod
    def _parse_problems(cls, problems: str) -> Mapping[str, MatchSpec]:
        """
        Problems can signal either unsatisfiability or unavailability.
        First will raise LibmambaUnsatisfiableError.
        Second will raise PackagesNotFoundError.

        Libmamba can return spec strings in two formats:
        - With dashes, e.g. package-1.2.3-h5487548_0
        - à la conda-build, e.g. package 1.2.*
        - just names, e.g. package
        """
        conflicts = []
        not_found = []
        problem_lines = problems.splitlines()[1:]
        for line in problem_lines:
            line = line.strip()
            words = line.split()
            if not line.startswith("- "):
                continue
            if "none of the providers can be installed" in line:
                if words[1] != "package" or words[3] != "requires":
                    raise ValueError(f"Unknown message: {line}")
                conflicts.append(cls._str_to_matchspec(words[2]))
                end = words.index("but")
                conflicts.append(cls._str_to_matchspec(words[4:end]))
            elif "- nothing provides" in line:
                marker = next((i for (i, w) in enumerate(words) if w == "needed"), None)
                if marker:
                    conflicts.append(cls._str_to_matchspec(words[-1]))
                start = 3 if marker == 4 else 4
                not_found.append(cls._str_to_matchspec(words[start:marker]))
            elif "has constraint" in line and "conflicting with" in line:
                # package libzlib-1.2.11-h4e544f5_1014 has constraint zlib 1.2.11 *_1014
                # conflicting with zlib-1.2.13-h998d150_0
                conflicts.append(cls._str_to_matchspec(words[-1]))
            elif "cannot install both pin-" in line and "and pin-" in line:
                # a pin is in conflict with another pin
                pin_a = words[3].rsplit("-", 1)[0]
                pin_b = words[5].rsplit("-", 1)[0]
                conflicts.append(MatchSpec(pin_a))
                conflicts.append(MatchSpec(pin_b))
            elif "is excluded by strict repo priority" in line:
                # package python-3.7.6-h0371630_2 is excluded by strict repo priority
                conflicts.append(cls._str_to_matchspec(words[2]))
            else:
                log.debug("! Problem line not recognized: %s", line)

        return {
            "conflicts": {s.name: s for s in conflicts},
            "not_found": {s.name: s for s in not_found},
        }

    def _maybe_raise_for_problems(
        self,
        problems: Optional[Union[str, Mapping]] = None,
        previous_conflicts: Mapping[str, MatchSpec] = None,
        pins: Mapping[str, MatchSpec] = None,
        channels: Iterable[Channel] = (),
    ):
        if self.solver is None:
            raise RuntimeError("Solver is not initialized. Call `._setup_solver()` first.")

        if problems is None:
            problems = self.solver.problems_to_str()
        if isinstance(problems, str):
            parsed_problems = self._parse_problems(problems)

        # We allow conda-build (if present) to process the exception early
        self._maybe_raise_for_conda_build(
            {**parsed_problems["conflicts"], **parsed_problems["not_found"]},
            message=self._prepare_problems_message(),
        )

        unsatisfiable = parsed_problems["conflicts"]
        not_found = parsed_problems["not_found"]
        if not unsatisfiable and not_found:
            log.debug(
                "Inferred PackagesNotFoundError %s from conflicts:\n%s",
                tuple(not_found.keys()),
                problems,
            )
            # This is not a conflict, but a missing package in the channel
            exc = PackagesNotFoundError(
                tuple(not_found.values()), tuple(channels or self.channels)
            )
            exc.allow_retry = False
            raise exc

        previous = previous_conflicts or {}
        previous_set = set(previous.values())
        current_set = set(unsatisfiable.values())

        diff = current_set.difference(previous_set)
        if len(diff) > 1 and "python" in unsatisfiable:
            # Only report python as conflict if it's the only conflict reported
            # This helps us prioritize neutering for other dependencies first
            unsatisfiable.pop("python")

        if (previous and (previous_set == current_set)) or len(diff) >= 10:
            # We have same or more (up to 10) unsatisfiable now! Abort to avoid recursion
            message = self._prepare_problems_message(pins=pins)
            exc = LibMambaUnsatisfiableError(message)
            # do not allow conda.cli.install to try more things
            exc.allow_retry = False
            raise exc
        return unsatisfiable

    def _prepare_problems_message(self, pins=None):
        legacy_errors = self.solver.problems_to_str()
        if " - " not in legacy_errors:
            # This makes 'explain_problems()' crash. Anticipate.
            return "Failed with empty error message."
        if "unsupported request" in legacy_errors:
            # This error makes 'explain_problems()' crash. Anticipate.
            log.info("Failed to explain problems. Unsupported request.")
            return legacy_errors
        if "is excluded by strict repo priority" in legacy_errors:
            # This will cause a lot of warnings until implemented in detail explanations
            log.info("Skipping error explanation. Excluded by strict repo priority.")
            return legacy_errors
        try:
            explained_errors = self.solver.explain_problems()
        except Exception as exc:
            log.warning("Failed to explain problems", exc_info=exc)
            return self._explain_with_pins(legacy_errors, pins)
        else:
            msg = f"{legacy_errors}\n{explained_errors}"
            return self._explain_with_pins(msg, pins)

    def _explain_with_pins(self, message, pins):
        """
        Add info about pins to the error message.
        This might be temporary as libmamba improves their error messages.
        As of 1.5.0, if a pin introduces a conflict, it results in a cryptic error message like:

        ```
        Encountered problems while solving:
          - cannot install both pin-1-1 and pin-1-1

        Could not solve for environment specs
        The following packages are incompatible
        └─ pin-1 is installable with the potential options
            ├─ pin-1 1, which can be installed;
            └─ pin-1 1 conflicts with any installable versions previously reported.
        ```

        Since the pin-N name is masked, we add the following snippet underneath:

        ```
        Pins seem to be involved in the conflict. Currently pinned specs:
         - python 2.7.* (labeled as 'pin-1')

        If Python is involved, try adding it explicitly to the command-line.
        ```
        """
        if pins and "pin-1" in message:  # add info about pins for easier debugging
            pin_message = "Pins seem to be involved in the conflict. Currently pinned specs:\n"
            for pin_name, spec in pins.items():
                pin_message += f" - {spec} (labeled as '{pin_name}')\n"
            return f"{message}\n\n{pin_message}"
        return message

    def _maybe_raise_for_conda_build(
        self,
        conflicting_specs: Mapping[str, MatchSpec],
        message: str = None,
    ):
        # TODO: Remove this hack for conda-build compatibility >_<
        # conda-build expects a slightly different exception format
        # good news is that we don't need to retry much, because all
        # conda-build envs are fresh - if we found a conflict, we report
        # right away to let conda build handle it
        if not self._called_from_conda_build():
            return

        from .conda_build_exceptions import ExplainedDependencyNeedsBuildingError

        # the patched index should contain the arch we are building this env for
        # if the index is empty, we default to whatever platform we are running on
        subdir = next((subdir for subdir in self.subdirs if subdir != "noarch"), context.subdir)
        exc = ExplainedDependencyNeedsBuildingError(
            packages=list(conflicting_specs.keys()),
            matchspecs=list(conflicting_specs.values()),
            subdir=subdir,
            explanation=message,
        )
        raise exc

    def _export_solved_records(
        self,
        in_state: SolverInputState,
        out_state: SolverOutputState,
        index: LibMambaIndexHelper,
    ):
        if self.solver is None:
            raise RuntimeError("Solver is not initialized. Call `._setup_solver()` first.")

        transaction = api.Transaction(
            index._pool,
            self.solver,
            api.MultiPackageCache(context.pkgs_dirs),
        )
        (names_to_add, names_to_remove), to_link, to_unlink = transaction.to_conda()

        for _, filename in to_unlink:
            for name, record in in_state.installed.items():
                if record.is_unmanageable:
                    # ^ Do not try to unlink virtual pkgs, virtual eggs, etc
                    continue
                if record.fn == filename:  # match!
                    out_state.records.pop(name, None, reason="Unlinked by solver")
                    break
            else:
                log.warn("Tried to unlink %s but it is not installed or manageable?", filename)

        for channel, filename, json_payload in to_link:
            record = self._package_record_from_json_payload(index, channel, filename, json_payload)
            # We need this check below to make sure noarch package get reinstalled
            # record metadata coming from libmamba is incomplete and won't pass the
            # noarch checks -- to fix it, we swap the metadata-only record with its locally
            # installed counterpart (richer in info)
            already_installed_record = in_state.installed.get(record.name)
            if (
                already_installed_record
                and record.subdir == "noarch"
                and already_installed_record.subdir == "noarch"
                and record.version == already_installed_record.version
                and record.build == already_installed_record.build
            ):
                # Replace repodata-only record with local-info-rich record counterpart
                record = already_installed_record

            out_state.records.set(
                record.name, record, reason="Part of solution calculated by libmamba"
            )

        # Fixes conda-build tests/test_api_build.py::test_croot_with_spaces
        if on_win and self._called_from_conda_build():
            for record in out_state.records.values():
                record.channel.location = percent_decode(record.channel.location)
                record.channel.name = percent_decode(record.channel.name)

    def _package_record_from_json_payload(
        self, index: LibMambaIndexHelper, channel: str, pkg_filename: str, json_payload: str
    ) -> PackageRecord:
        """
        The libmamba transactions cannot return full-blown objects from the C/C++ side.
        Instead, it returns the instructions to build one on the Python side:

        channel_info: dict
            Channel data, as built in .index.LibmambaIndexHelper._fetch_channel()
            This is retrieved from the .index._index mapping, keyed by channel URLs
        pkg_filename: str
            The filename (.tar.bz2 or .conda) of the selected record.
        json_payload: str
            A str-encoded JSON payload with the PackageRecord kwargs.
        """
        # conda-lock will inject virtual packages, but these are not in the index
        if pkg_filename.startswith("__") and "/@/" in channel:
            return PackageRecord(**json.loads(json_payload))

        kwargs = json.loads(json_payload)
        try:
            channel_info = index.get_info(channel)
        except KeyError:
            # this channel was never used to build the index, which
            # means we obtained an already installed PackageRecord
            # whose metadata contains a channel that doesn't exist
            pd = PrefixData(self.prefix)
            record = pd.get(kwargs["name"])
            if record and record.fn == pkg_filename:
                return record

        # Otherwise, these are records from the index
        kwargs["fn"] = pkg_filename
        kwargs["channel"] = channel_info.channel
        kwargs["url"] = join_url(channel_info.full_url, pkg_filename)
        if not kwargs.get("subdir"):  # missing in old channels
            kwargs["subdir"] = channel_info.channel.subdir
        if kwargs["subdir"] == "noarch":
            # libmamba doesn't keep 'noarch' type around, so infer for now
            if any(dep.split()[0] in ("python", "pypy") for dep in kwargs.get("depends", ())):
                kwargs["noarch"] = "python"
            else:
                kwargs["noarch"] = "generic"
        return PackageRecord(**kwargs)

    def _check_spec_compat(self, match_spec: Union[MatchSpec, None]) -> Union[MatchSpec, None]:
        """
        Make sure we are not silently ingesting MatchSpec fields we are not
        doing anything with!

        TODO: We currently allow `subdir` but we are not handling it right now.
        """
        if match_spec is None:
            return None
        supported = "name", "version", "build", "channel", "subdir"
        unsupported_but_set = []
        for field in match_spec.FIELD_NAMES:
            value = match_spec.get_raw_value(field)
            if value and field not in supported:
                unsupported_but_set.append(field)
        if unsupported_but_set:
            raise InvalidMatchSpec(
                match_spec,
                "Libmamba only supports a subset of the MatchSpec interface for now. "
                f"You can only use {supported}, but you tried to use "
                f"{tuple(unsupported_but_set)}.",
            )
        if (
            match_spec.get_raw_value("channel") == "defaults"
            and context.default_channels == DEFAULT_CHANNELS
        ):
            # !!! Temporary !!!
            # Apply workaround for defaults::pkg-name specs.
            # We need to replace it with the actual channel name (main, msys2, r)
            # Instead of searching in the index, we apply a simple heuristic:
            # - R packages are [_]r-*, mro-*, rpy or rstudio
            # - Msys2 packages are m2-*, m2w64-*, or msys2-*
            # - Everything else is in main
            name = match_spec.name.lower()
            if name in ("r", "rpy2", "rstudio") or name.startswith(("r-", "_r-", "mro-")):
                channel = "pkgs/r"
            elif name.startswith(("m2-", "m2w64-", "msys2-")):
                channel = "pkgs/msys2"
            else:
                channel = "pkgs/main"
            match_spec = MatchSpec(match_spec, channel=channel)

        return match_spec

    def _reset(self):
        self.solver = None
        self._solver_options = None

    def _called_from_conda_build(self):
        """
        conda build calls the solver via `conda.plan.install_actions`, which
        overrides Solver._index (populated in the classic solver, but empty for us)
        with a custom index. We can use this to detect whether conda build is in use
        and apply some compatibility fixes.
        """
        return (
            # conda_build.environ.get_install_actions will always pass a custom 'index'
            # which conda.plan.install_actions uses to override our null Solver._index
            getattr(self, "_index", None)
            # Is conda build in use? In that case, it should have been imported
            and "conda_build" in sys.modules
            # Confirm conda_build.environ's 'get_install_actions' and conda.plan's
            # 'install_actions' are in the call stack. We don't check order or
            # contiguousness, but what are the chances at this point...?
            # frame[3] contains the name of the function in that frame of the stack
            and {"install_actions", "get_install_actions"} <= {frame[3] for frame in stack()}
        )

    def _notify_conda_outdated(
        self,
        link_precs,
        index: LibMambaIndexHelper = None,
        final_state: Iterable[PackageRecord] = None,
    ):
        """
        We are overriding the base class implementation, which gets called in
        Solver.solve_for_diff() once 'link_precs' is available. However, we
        are going to call it before (in .solve_final_state(), right after the solve).
        That way we can reuse the IndexHelper and SolverOutputState instances we have
        around, which contains the channel and env information we need, before losing them.
        """
        if index is None and final_state is None:
            # The parent class 'Solver.solve_for_diff()' method will call this method again
            # with only 'link_precs' as the argument, because that's the original method signature.
            # We have added two optional kwargs (index and final_state) so we can call this method
            # earlier, in .solve_final_state(), while we still have access to the index helper
            # (which allows us to query the available packages in the channels quickly, without
            # reloading the channels with conda) and the final_state (which gives the list of
            # packages to be installed). So, if both index and final_state are None, we return
            # because that means that the method is being called from .solve_for_diff() and at
            # that point we will have already called it from .solve_for_state().
            return
        if not context.notify_outdated_conda or context.quiet:
            # This check can be silenced with a specific option in the context or in quiet mode
            return

        # manually check base prefix since `PrefixData(...).get("conda", None) is expensive
        # once prefix data is lazy this might be a different situation
        current_conda_prefix_rec = None
        conda_meta_prefix_directory = os.path.join(context.conda_prefix, "conda-meta")
        with suppress(OSError, ValueError):
            if os.path.lexists(conda_meta_prefix_directory):
                for entry in os.scandir(conda_meta_prefix_directory):
                    if (
                        entry.is_file()
                        and entry.name.endswith(".json")
                        and entry.name.rsplit("-", 2)[0] == "conda"
                    ):
                        with open(entry.path) as f:
                            current_conda_prefix_rec = PrefixRecord(**json.loads(f.read()))
                        break
        if not current_conda_prefix_rec:
            # We are checking whether conda can be found in the environment conda is
            # running from. Unless something is really wrong, this should never happen.
            return

        channel_name = current_conda_prefix_rec.channel.canonical_name
        if channel_name in (UNKNOWN_CHANNEL, "@", "<develop>", "pypi"):
            channel_name = "defaults"

        # only check the loaded index if it contains the channel conda should come from
        # otherwise ignore
        index_channels = {getattr(chn, "canonical_name", chn) for chn in index._channels}
        if channel_name not in index_channels:
            return

        # we only want to check if a newer conda is available in the channel we installed it from
        conda_newer_str = f"{channel_name}::conda>{_conda_version}"
        conda_newer_spec = MatchSpec(conda_newer_str)

        # if target prefix is the same conda is running from
        # maybe the solution we are proposing already contains
        # an updated conda! in that case, we don't need to check further
        if paths_equal(self.prefix, context.conda_prefix):
            if any(conda_newer_spec.match(record) for record in final_state):
                return

        # check if the loaded index contains records that match a more recent conda version
        conda_newer_records = index.search(conda_newer_str)

        # print instructions to stderr if we found a newer conda
        if conda_newer_records:
            newest = max(conda_newer_records, key=lambda x: VersionOrder(x.version))
            print(
                dedent(
                    f"""

                    ==> WARNING: A newer version of conda exists. <==
                        current version: {_conda_version}
                        latest version: {newest.version}

                    Please update conda by running

                        $ conda update -n base -c {channel_name} conda

                    """
                ),
                file=sys.stderr,
            )
