# Copyright (C) 2024 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
This module defines the conda.core.solve.Solver interface and its immediate helpers
"""

from __future__ import annotations

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
from typing import Iterable, Mapping, Optional, Sequence, TYPE_CHECKING, List, Tuple

from boltons.setutils import IndexedSet
from conda import __version__ as _conda_version
from conda.base.constants import (
    DEFAULT_CHANNELS,
    REPODATA_FN,
    UNKNOWN_CHANNEL,
    ChannelPriority,
    DepsModifier,
    UpdateModifier,
)
from conda.base.context import context
from conda.common.compat import on_win
from conda.common.constants import NULL
from conda.common.io import Spinner, timeout, time_recorder
from conda.common.path import paths_equal
from conda.common.url import join_url, percent_decode
from conda.core.package_cache_data import PackageCacheData
from conda.core.prefix_data import PrefixData
from conda.core.solve import Solver
from conda.exceptions import (
    CondaValueError,
    InvalidMatchSpec,
    InvalidSpec,
    PackagesNotFoundError,
    ParseError,
    UnsatisfiableError,
)
from conda.models.channel import Channel
from conda.models.match_spec import MatchSpec
from conda.models.records import PackageRecord, PrefixRecord
from conda.models.version import VersionOrder
from libmambapy.solver import Request, Solution
from libmambapy.solver.libsolv import Solver as LibsolvSolver, UnSolvable
from libmambapy.specs import MatchSpec as LibmambaMatchSpec, NoArchType, PackageInfo

from . import __version__
from .exceptions import LibMambaUnsatisfiableError
from .index2 import LibMambaIndexHelper
from .mamba_utils import mamba_version
from .state import SolverInputState, SolverOutputState
from .utils import is_channel_available

if TYPE_CHECKING:
    from conda.auxlib import _Null

log = logging.getLogger(f"conda.{__name__}")


class LibMambaSolver(Solver):
    MAX_SOLVER_ATTEMPTS_CAP = 10
    _uses_ssc = False

    @staticmethod
    @lru_cache(maxsize=None)
    def user_agent():
        """
        Expose this identifier to allow conda to extend its user agent if required
        """
        return f"conda-libmamba-solver/{__version__} libmambapy/{mamba_version()}"

    def __init__(
        self,
        prefix: os.PathLike,
        channels: Iterable[Channel | str],
        subdirs: Iterable[str] = (),
        specs_to_add: Iterable[MatchSpec | str] = (),
        specs_to_remove: Iterable[MatchSpec | str] = (),
        repodata_fn: str = REPODATA_FN,
        command: str | _Null = NULL,
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

        self._repodata_fn = self._maybe_ignore_current_repodata()

    def solve_final_state(
        self,
        update_modifier: UpdateModifier | _Null = NULL,
        deps_modifier: DepsModifier | _Null = NULL,
        prune: bool | _Null = NULL,
        ignore_pinned: bool | _Null = NULL,
        force_remove: bool | _Null = NULL,
        should_retry_solve: bool = False,
    ) -> IndexedSet[PackageRecord]:
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
        maybe_final_state = out_state.early_exit()
        if maybe_final_state is not None:
            return maybe_final_state

        channels = self._collect_channel_list(in_state)
        conda_build_channels = self._collect_channel_list_conda_build()
        with Spinner(
            self._collect_all_metadata_spinner_message(channels, conda_build_channels),
            enabled=not context.verbosity and not context.quiet,
            json=context.json,
        ):
            index = self._collect_all_metadata(
                channels=channels,
                conda_build_channels=conda_build_channels,
                in_state=in_state,
            )

        with Spinner(
            self._solving_loop_spinner_message(),
            enabled=not context.verbosity and not context.quiet,
            json=context.json,
        ):
            # This function will copy and mutate `out_state`
            # Make sure we get the latest copy to return the correct solution below
            out_state = self._solving_loop(in_state, out_state, index)
            self.neutered_specs = tuple(out_state.neutered.values())
            solution = out_state.current_solution

        # Check whether conda can be updated; this is normally done in .solve_for_diff()
        # but we are doing it now so we can reuse the index
        self._notify_conda_outdated(None, index, solution)

        return solution

    # region Metadata collection
    ###########################

    def _collect_all_metadata_spinner_message(
        self,
        channels: Iterable[Channel],
        conda_build_channels: Iterable[Channel | str] = (),
    ) -> str:
        if self._called_from_conda_build():
            msg = "Reloading output folder"
            if conda_build_channels:
                names = list(
                    dict.fromkeys([Channel(c).canonical_name for c in conda_build_channels])
                )
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

    def _collect_channel_list(self, in_state: SolverInputState) -> List[Channel]:
        channels = [*self.channels, *in_state.channels_from_specs()]

        # TODO: Deprecate 'channels_from_installed' functionality?
        override = (getattr(context, "_argparse_args", None) or {}).get("override_channels")
        if not os.getenv("CONDA_LIBMAMBA_SOLVER_NO_CHANNELS_FROM_INSTALLED") and not override:
            # see https://github.com/conda/conda-libmamba-solver/issues/108
            all_urls_so_far = [url for c in channels for url in Channel(c).urls(False)]
            installed_channels = in_state.channels_from_installed(seen=all_urls_so_far)
            for channel in installed_channels:
                # Only add to list if resource is available; check has timeout=1s
                if timeout(1, is_channel_available, channel.base_url, default_return=False):
                    channels.append(channel)

        channels.extend(in_state.maybe_free_channel())
        return channels

    def _collect_channel_list_conda_build(self) -> List[Channel]:
        if self._called_from_conda_build():
            # We need to recover the local dirs (conda-build's local, output_folder, etc)
            # from the index. This is a bit of a hack, but it works.
            conda_build_channels = {
                rec.channel: None for rec in self._index if rec.channel.scheme == "file"
            }
            return list(conda_build_channels)
        return []

    @time_recorder(module_name=__name__)
    def _collect_all_metadata(
        self,
        channels: Iterable[Channel],
        conda_build_channels: Iterable[Channel],
        in_state: SolverInputState,
    ) -> LibMambaIndexHelper:
        index = LibMambaIndexHelper(
            channels=[*conda_build_channels, *channels],
            repodata_fn=self._repodata_fn,
            installed_records=(
                *in_state.installed.values(),
                *in_state.virtual.values(),
            ),
            pkgs_dirs=context.pkgs_dirs if context.offline else (),
        )
        for channel in conda_build_channels:
            index.reload_channel(channel)
        return index

    # endregion

    # region Solving
    ###############

    def _solving_loop_spinner_message(self) -> str:
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

    @time_recorder(module_name=__name__)
    def _solving_loop(
        self,
        in_state: SolverInputState,
        out_state: SolverOutputState,
        index: LibMambaIndexHelper,
    ) -> IndexedSet[PackageRecord]:
        for attempt in range(1, self._max_attempts(in_state) + 1):
            solved, outcome = self._solve_attempt(in_state, out_state, index, attempt=attempt)
            if solved:
                return self._export_solution(out_state, outcome)

    def _solve_attempt(
        self,
        in_state: SolverInputState,
        out_state: SolverOutputState,
        index: LibMambaIndexHelper,
        attempt: int = 1,
    ) -> tuple[bool, Solution | UnSolvable]:
        log.info("Solver attempt: #%d", attempt)
        flags = self._solver_flags()
        jobs = self._specs_to_request_jobs(in_state, out_state)
        request = Request(jobs=jobs, flags=flags)
        solver = LibsolvSolver()
        outcome = solver.solve(index.db, request)
        if isinstance(outcome, Solution):
            out_state.conflicts.clear()
            return True, outcome
        problems = outcome.explain_problems()
        return False, outcome

    def _solver_flags(self) -> Request.Flags:
        flags = Request.Flags()
        flags.allow_downgrade = True
        if context.channel_priority is ChannelPriority.STRICT:
            flags.strict_repo_priority = True
        if self.specs_to_remove and self._command in ("remove", None, NULL):
            flags.allow_uninstall = True
        return flags

    def _specs_to_request_jobs(
        self,
        in_state: SolverInputState,
        out_state: SolverOutputState,
    ) -> list[Request.Job]:
        if in_state.is_removing:
            jobs = self._specs_to_request_jobs_remove(in_state, out_state)
        elif self._called_from_conda_build():
            jobs = self._specs_to_request_jobs_conda_build(in_state, out_state)
        else:
            jobs = self._specs_to_request_jobs_add(in_state, out_state)

        request_jobs = []
        json_friendly = {}
        for (task_name, task_type), specs in jobs.items():
            for spec in specs:
                request_jobs.append(task_type(spec))
                json_friendly.setdefault(task_name, []).append(str(spec))
        json_str = json.dumps(json_friendly, indent=2)
        log.info("The solver will handle these requests:\n%s", json_str)
        return request_jobs

    def _specs_to_request_jobs_add(self, in_state: SolverInputState, out_state: SolverOutputState):
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

        # Job types
        ADD_PIN = "ADD_PIN", Request.Pin
        INSTALL = "INSTALL", Request.Install
        UPDATE = "UPDATE", Request.Update
        # ALLOW_UNINSTALL = "ALLOW_UNINSTALL", api.SOLVER_ALLOWUNINSTALL
        USERINSTALLED = "USERINSTALLED", Request.Keep
        LOCK = "LOCK", Request.Freeze

        for name in out_state.specs:
            if name.startswith("__"):
                continue  # ignore virtual packages
            installed: PackageRecord = in_state.installed.get(name)
            if installed:
                installed_libmamba_spec = self._conda_spec_to_libmamba_spec(
                    self._check_spec_compat(installed.to_match_spec())
                )
            else:
                installed_libmamba_spec = None
            requested: MatchSpec = self._check_spec_compat(in_state.requested.get(name))
            history: MatchSpec = self._check_spec_compat(in_state.history.get(name))
            pinned: MatchSpec = self._check_spec_compat(in_state.pinned.get(name))
            conflicting: MatchSpec = self._check_spec_compat(out_state.conflicts.get(name))

            if name in user_installed and not in_state.prune and not conflicting:
                tasks[USERINSTALLED].append(installed_libmamba_spec)

            # These specs are explicit in some sort of way
            if pinned and not pinned.is_name_only_spec:
                # these are the EXPLICIT pins; conda also uses implicit pinning to
                # constrain updates too but those can be overridden in case of conflicts.
                # name-only pins are treated as locks when installed, see below
                tasks[ADD_PIN].append(self._conda_spec_to_libmamba_spec(pinned))
            # in libmamba, pins and installs are compatible tasks (pin only constrains,
            # does not 'request' a package). In classic, pins were actually targeted installs
            # so they were exclusive
            if requested:
                if requested.is_name_only_spec and pinned and not pinned.is_name_only_spec:
                    # for name-only specs, this is a no-op; we already added the pin above
                    # but we will constrain it again in the install task to have better
                    # error messages if not solvable
                    libmamba_spec = self._conda_spec_to_libmamba_spec(pinned)
                else:
                    libmamba_spec = self._conda_spec_to_libmamba_spec(requested)
                if installed:
                    tasks[UPDATE].append(libmamba_spec)
                    # tasks[ALLOW_UNINSTALL].append(name)
                else:
                    tasks[INSTALL].append(libmamba_spec)
            elif name in in_state.always_update:
                tasks[UPDATE].append(name)
                # tasks[ALLOW_UNINSTALL].append(name)
            # These specs are "implicit"; the solver logic massages them for better UX
            # as long as they don't cause trouble
            elif in_state.prune:
                continue
            elif name == "python" and installed and not pinned:
                pyver = ".".join(installed.version.split(".")[:2])
                tasks[ADD_PIN].append(f"python {pyver}.*")
            elif history:
                if conflicting and history.strictness == 3:
                    # relax name-version-build (strictness=3) history specs that cause conflicts
                    # this is called neutering and makes test_neutering_of_historic_specs pass
                    spec = f"{name} {history.version}.*" if history.version else name
                    tasks[INSTALL].append(spec)
                else:
                    tasks[INSTALL].append(self._conda_spec_to_libmamba_spec(history))
            elif installed:
                if conflicting:
                    ...
                    # tasks[ALLOW_UNINSTALL].append(name)
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
                        tasks[LOCK].append(installed_libmamba_spec)

        # Sort tasks by priority
        # This ensures that more important tasks are added to the solver first
        returned_tasks = {}
        for task_type in (
            ADD_PIN,
            INSTALL,
            UPDATE,
            # ALLOW_UNINSTALL,
            USERINSTALLED,
            LOCK,
        ):
            if task_type in tasks:
                returned_tasks[task_type] = tasks[task_type]
        return returned_tasks

    def _specs_to_request_jobs_remove(
        self, in_state: SolverInputState, out_state: SolverOutputState
    ):
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
        ERASE = ("ERASE | CLEANDEPS", api.SOLVER_ERASE | api.SOLVER_CLEANDEPS)
        for name, spec in in_state.requested.items():
            spec = self._check_spec_compat(spec)
            tasks[ERASE].append(str(spec))

        return dict(tasks)

    def _specs_to_request_jobs_conda_build(
        self, in_state: SolverInputState, out_state: SolverOutputState
    ):
        tasks = defaultdict(list)
        INSTALL = "INSTALL", api.SOLVER_INSTALL
        for name, spec in in_state.requested.items():
            if name.startswith("__"):
                continue
            spec = self._check_spec_compat(spec)
            spec = self._fix_version_field_for_conda_build(spec)
            tasks[INSTALL].append(spec.conda_build_form())

        return dict(tasks)

    def _export_solution(self, out_state: SolverOutputState, solution: Solution):
        for action in solution.actions:
            record_to_install: PackageInfo = getattr(action, "install", None)
            record_to_remove: PackageInfo = getattr(action, "remove", None)
            if record_to_install:
                if record_to_install.name.startswith("__"):
                    continue
                record = self._package_info_to_package_record(record_to_install)
                out_state.records[record.name] = record
            elif record_to_remove:
                if record_to_remove.name.startswith("__"):
                    continue
                record = self._package_info_to_package_record(record_to_remove)
                out_state.records.pop(record.name, None)
        return out_state

    def _package_info_to_package_record(self, pkg: PackageInfo) -> PackageRecord:
        if pkg.noarch == NoArchType.Python:
            noarch = "python"
        elif pkg.noarch == NoArchType.Generic:
            noarch = "generic"
        else:
            noarch = None
        return PackageRecord(
            name=pkg.name,
            version=pkg.version,
            build=pkg.build_string,
            build_number=pkg.build_number,
            channel=pkg.channel,
            url=pkg.package_url,
            subdir=pkg.platform,
            fn=pkg.filename,
            license=pkg.license,
            md5=pkg.md5,
            sha256=pkg.sha256,
            signatures=pkg.signatures,
            track_features=pkg.track_features,
            depends=pkg.dependencies,
            constrains=pkg.constrains,
            defaulted_keys=pkg.defaulted_keys,
            noarch=noarch,
            size=pkg.size,
            timestamp=pkg.timestamp,
        )

    # endregion

    # region General helpers
    #######################

    def _log_info(self):
        log.info("conda version: %s", _conda_version)
        log.info("conda-libmamba-solver version: %s", __version__)
        log.info("libmambapy version: %s", mamba_version())
        log.info("Target prefix: %r", self.prefix)
        log.info("Command: %s", sys.argv)

    def _called_from_conda_build(self) -> bool:
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

    def _check_spec_compat(self, spec: MatchSpec | None) -> MatchSpec | None:
        return spec

    def _conda_spec_to_libmamba_spec(self, spec: MatchSpec) -> LibmambaMatchSpec:
        return LibmambaMatchSpec.parse(str(spec))

    def _maybe_ignore_current_repodata(self) -> str:
        is_repodata_fn_set = False
        for config in context.collect_all().values():
            for key, value in config.items():
                if key == "repodata_fns" and value:
                    is_repodata_fn_set = True
                    break
        if self._repodata_fn == "current_repodata.json" and not is_repodata_fn_set:
            log.debug(
                "Ignoring repodata_fn='current_repodata.json', defaulting to %s",
                REPODATA_FN,
            )
            return REPODATA_FN
        return self._repodata_fn

    def _max_attempts(self, in_state: SolverInputState, default: int = 1):
        from_env_var = os.environ.get("CONDA_LIBMAMBA_SOLVER_MAX_ATTEMPTS")
        installed_count = len(in_state.installed)
        if from_env_var:
            try:
                max_attempts_from_env = int(from_env_var)
            except ValueError:
                raise CondaValueError(
                    f"CONDA_LIBMAMBA_SOLVER_MAX_ATTEMPTS='{from_env_var}'. Must be int."
                )
            if max_attempts_from_env < 1:
                raise CondaValueError(
                    f"CONDA_LIBMAMBA_SOLVER_MAX_ATTEMPTS='{max_attempts_from_env}'. Must be >=1."
                )
            elif max_attempts_from_env > installed_count:
                log.warning(
                    "CONDA_LIBMAMBA_SOLVER_MAX_ATTEMPTS='%s' is higher than the number of "
                    "installed packages (%s). Using that one instead.",
                    max_attempts_from_env,
                    installed_count,
                )
                return installed_count
            else:
                return max_attempts_from_env
        elif in_state.update_modifier.FREEZE_INSTALLED and installed_count:
            # this the default, but can be overriden with --update-specs
            # we cap at MAX_SOLVER_ATTEMPTS_CAP attempts to avoid things
            # getting too slow in large environments
            return min(self.MAX_SOLVER_ATTEMPTS_CAP, installed_count)
        else:
            return default

    def _notify_conda_outdated(
        self,
        link_precs: Iterable[PackageRecord],
        index: LibMambaIndexHelper | None = None,
        final_state: Iterable[PackageRecord] | None = None,
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

    # endregion
