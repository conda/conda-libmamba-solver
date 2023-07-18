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
from functools import lru_cache
from inspect import stack
from textwrap import dedent
from typing import Iterable, Mapping, Optional, Sequence, Union

import libmambapy as api
from boltons.setutils import IndexedSet
from conda import __version__ as _conda_version
from conda.base.constants import (
    REPODATA_FN,
    UNKNOWN_CHANNEL,
    ChannelPriority,
    DepsModifier,
    UpdateModifier,
    on_win,
)
from conda.base.context import context
from conda.common.constants import NULL
from conda.common.io import Spinner
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
from conda.models.records import PackageRecord
from conda.models.version import VersionOrder

from . import __version__
from .exceptions import LibMambaUnsatisfiableError
from .index import LibMambaIndexHelper, _CachedLibMambaIndexHelper
from .mamba_utils import init_api_context, mamba_version
from .state import SolverInputState, SolverOutputState

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

        if os.getenv("CONDA_LIBMAMBA_SOLVER_NO_CHANNELS_FROM_INSTALLED"):
            # see https://github.com/conda/conda-libmamba-solver/issues/108
            channels_from_installed = ()
        else:
            channels_from_installed = in_state.channels_from_installed()

        all_channels = (
            *conda_bld_channels,
            *self.channels,
            *in_state.channels_from_specs(),
            *channels_from_installed,
            *in_state.maybe_free_channel(),
        )
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
                message = self._prepare_problems_message()
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
            (api.SOLVER_FLAG_ALLOW_UNINSTALL, 1),
            (api.SOLVER_FLAG_INSTALL_ALSO_UPDATES, 1),
            (api.SOLVER_FLAG_FOCUS_BEST, 1),
            (api.SOLVER_FLAG_BEST_OBEY_POLICY, 1),
        ]
        if context.channel_priority is ChannelPriority.STRICT:
            solver_options.append((api.SOLVER_FLAG_STRICT_REPO_PRIORITY, 1))

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
        tasks = self._specs_to_tasks(in_state, out_state)
        for (task_name, task_type), specs in tasks.items():
            log.debug("Adding task %s with specs %s", task_name, specs)
            self.solver.add_jobs(specs, task_type)

        # ## Run solver
        solved = self.solver.solve()

        if solved:
            out_state.conflicts.clear(reason="Solution found")
            return solved

        problems = self.solver.problems_to_str()
        old_conflicts = out_state.conflicts.copy()
        new_conflicts = self._maybe_raise_for_problems(problems, old_conflicts)
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
        # These packages receive special protection, since they will be
        # exempt from conflict treatment (ALLOWUNINSTALL) and if installed
        # their updates will be considered ESSENTIAL and USERINSTALLED
        protected = (
            ["python", "conda"]
            + list(in_state.history.keys())
            + list(in_state.aggressive_updates.keys())
        )

        # Fast-track python version changes
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

        tasks = defaultdict(list)
        for name, spec in out_state.specs.items():
            if name.startswith("__"):
                continue
            spec = self._check_spec_compat(spec)
            spec_str = self._spec_to_str(spec)
            installed = in_state.installed.get(name)

            key = "INSTALL", api.SOLVER_INSTALL

            # Fast-track Python version changes: mark non-noarch Python-depending packages as
            # conflicting (see `python_version_might_change` definition above for more details)
            if python_version_might_change and installed is not None:
                if installed.noarch is not None:
                    continue
                for dep in installed.depends:
                    dep_spec = MatchSpec(dep)
                    if dep_spec.name in ("python", "python_abi"):
                        reason = "Python version might change and this package depends on Python"
                        out_state.conflicts.update(
                            {name: spec},
                            reason=reason,
                            overwrite=False,
                        )
                        break

            # ## Low-prio task ###
            if name in out_state.conflicts and name not in protected:
                tasks[("DISFAVOR", api.SOLVER_DISFAVOR)].append(spec_str)
                tasks[("ALLOWUNINSTALL", api.SOLVER_ALLOWUNINSTALL)].append(spec_str)

            if installed is not None:
                # ## Regular task ###
                key = "UPDATE", api.SOLVER_UPDATE

                # ## Protect if installed AND history
                if name in protected:
                    installed_spec = self._spec_to_str(installed.to_match_spec())
                    tasks[("USERINSTALLED", api.SOLVER_USERINSTALLED)].append(installed_spec)
                    # This is "just" an essential job, so it gets higher priority in the solver
                    # conflict resolution. We do this because these are "protected" packages
                    # (history, aggressive updates) that we should try not messing with if
                    # conflicts appear
                    key = ("UPDATE | ESSENTIAL", api.SOLVER_UPDATE | api.SOLVER_ESSENTIAL)

                # ## Here we deal with the "bare spec update" problem
                # ## I am only adding this for legacy / test compliancy reasons; forced updates
                # ## like this should (imo) use constrained specs (e.g. conda install python=3)
                # ## or the update command as in `conda update python`. however conda thinks
                # ## differently of update vs install (quite counterintuitive):
                # ##   https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/installing-with-conda.html#conda-update-versus-conda-install  # noqa
                # ## this is tested in:
                # ##   tests/core/test_solve.py::test_pinned_1
                # ##   tests/test_create.py::IntegrationTests::test_update_with_pinned_packages
                # ## fixing this changes the outcome in other tests!
                # let's say we have an environment with python 2.6 and we say `conda install
                # python` libsolv will say we already have python and there's no reason to do
                # anything else even if we force an update with essential, other packages in the
                # environment (built for py26) will keep it in place. we offer two ways to deal
                # with this libsolv behaviour issue:
                #   A) introduce an artificial version spec `python !=<currently installed>`
                #   B) use FORCEBEST -- this would be ideal, but sometimes in gets in the way,
                #      so we only use it as a last attempt effort.
                # NOTE: This is a dirty-ish workaround... rethink?
                requested = in_state.requested.get(name)
                conditions = (
                    requested,
                    spec == requested,
                    spec.strictness == 1,
                    self._command in ("update", "update+last_solve_attempt", None, NULL),
                    in_state.deps_modifier != DepsModifier.ONLY_DEPS,
                    in_state.update_modifier
                    not in (UpdateModifier.UPDATE_DEPS, UpdateModifier.FREEZE_INSTALLED),
                )
                if all(conditions):
                    if "last_solve_attempt" in str(self._command):
                        key = (
                            "UPDATE | ESSENTIAL | FORCEBEST",
                            api.SOLVER_UPDATE | api.SOLVER_ESSENTIAL | api.SOLVER_FORCEBEST,
                        )
                    else:
                        # NOTE: This is ugly and there should be another way
                        spec_str = self._spec_to_str(
                            MatchSpec(spec, version=f">{installed.version}")
                        )

            tasks[key].append(spec_str)

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
        for line in problems.splitlines():
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

        return {
            "conflicts": {s.name: s for s in conflicts},
            "not_found": {s.name: s for s in not_found},
        }

    def _maybe_raise_for_problems(
        self,
        problems: Optional[Union[str, Mapping]] = None,
        previous_conflicts: Mapping[str, MatchSpec] = None,
    ):
        if self.solver is None:
            raise RuntimeError("Solver is not initialized. Call `._setup_solver()` first.")

        if problems is None:
            problems = self.solver.problems_to_str()
        if isinstance(problems, str):
            problems = self._parse_problems(problems)

        # We allow conda-build (if present) to process the exception early
        self._maybe_raise_for_conda_build(
            {**problems["conflicts"], **problems["not_found"]},
            message=self._prepare_problems_message(),
        )

        unsatisfiable = problems["conflicts"]
        not_found = problems["not_found"]
        if not unsatisfiable and not_found:
            # This is not a conflict, but a missing package in the channel
            exc = PackagesNotFoundError(not_found.values(), self.channels)
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
            message = self._prepare_problems_message()
            exc = LibMambaUnsatisfiableError(message)
            # do not allow conda.cli.install to try more things
            exc.allow_retry = False
            raise exc
        return unsatisfiable

    def _prepare_problems_message(self):
        legacy_errors = self.solver.problems_to_str()
        if "unsupported request" in legacy_errors:
            # This error makes 'explain_problems()' crash. Anticipate.
            log.info("Failed to explain problems. Unsupported request.")
            return legacy_errors
        try:
            explained_errors = self.solver.explain_problems()
        except Exception as exc:
            log.warning("Failed to explain problems", exc_info=exc)
            return legacy_errors
        else:
            return f"{legacy_errors}\n{explained_errors}"

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

        from .exceptions import ExplainedDependencyNeedsBuildingError

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

        channel_info = index.get_info(channel)
        kwargs = json.loads(json_payload)
        kwargs["fn"] = pkg_filename
        kwargs["channel"] = channel_info.channel
        kwargs["url"] = join_url(channel_info.full_url, pkg_filename)
        if not kwargs.get("subdir"):  # missing in old channels
            kwargs["subdir"] = channel_info.channel.subdir
        return PackageRecord(**kwargs)

    def _check_spec_compat(self, match_spec: MatchSpec) -> MatchSpec:
        """
        Make sure we are not silently ingesting MatchSpec fields we are not
        doing anything with!

        TODO: We currently allow `subdir` but we are not handling it right now.
        """
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
        if match_spec.get_raw_value("channel") == "defaults":
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

        current_conda_prefix_rec = PrefixData(context.conda_prefix).get("conda", None)
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
        conda_newer_str = f"{channel_name}::conda>{current_conda_prefix_rec.version}"
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
