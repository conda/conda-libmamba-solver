# Copyright (C) 2024 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
This module defines the conda.core.solve.Solver interface and its immediate helpers
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
from contextlib import suppress
from functools import lru_cache
from inspect import stack
from textwrap import dedent
from typing import TYPE_CHECKING

from conda import __version__ as _conda_version
from conda.base.constants import (
    REPODATA_FN,
    UNKNOWN_CHANNEL,
    ChannelPriority,
)
from conda.base.context import context
from conda.common.constants import NULL
from conda.common.io import Spinner, time_recorder, timeout
from conda.common.path import paths_equal
from conda.core.solve import Solver
from conda.exceptions import (
    CondaValueError,
    PackagesNotFoundError,
    UnsatisfiableError,
)
from conda.models.channel import Channel
from conda.models.match_spec import MatchSpec
from conda.models.records import PackageRecord, PrefixRecord
from conda.models.version import VersionOrder
from libmambapy import Palette
from libmambapy.solver import Request, Solution
from libmambapy.solver.libsolv import Solver as LibsolvSolver
from libmambapy.specs import MatchSpec as LibmambaMatchSpec
from libmambapy.specs import NoArchType

from . import __version__
from .exceptions import LibMambaUnsatisfiableError
from .index import LibMambaIndexHelper
from .mamba_utils import mamba_version
from .state import SolverInputState, SolverOutputState
from .utils import is_channel_available

if TYPE_CHECKING:
    from typing import Iterable, Mapping, Sequence

    from boltons.setutils import IndexedSet
    from conda.auxlib import _Null
    from conda.base.constants import (
        DepsModifier,
        UpdateModifier,
    )
    from libmambapy.solver.libsolv import Database, UnSolvable
    from libmambapy.specs import PackageInfo


_mamba_auto_palette = Palette.terminal() if sys.stdin.isatty() else Palette.no_color()
_mamba_nocolor_palette = Palette.no_color()
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
            out_state.check_for_pin_conflicts(index)

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
    ############################

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

    def _collect_channel_list(self, in_state: SolverInputState) -> list[Channel]:
        channels = [*self.channels]
        channels_by_name = {}
        for channel in channels:
            channels_by_name.setdefault(channel.name or channel.canonical_name, []).append(channel)
        for spec_channel in in_state.channels_from_specs():
            same_name = channels_by_name.get(spec_channel.name or spec_channel.canonical_name)
            if same_name:
                # do not add spec channel if there's already one with the same name
                continue
            channels.append(spec_channel)

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

    def _collect_channel_list_conda_build(self) -> list[Channel]:
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
            subdirs=self.subdirs,
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
    ################

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
            try:
                solved, outcome = self._solve_attempt(in_state, out_state, index, attempt=attempt)
                if solved:
                    break
            except (UnsatisfiableError, PackagesNotFoundError):
                solved = False
                break  # try with last attempt
            else:  # didn't solve yet, but can retry
                out_state = SolverOutputState(
                    solver_input_state=in_state,
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
                    if not record.is_unmanageable
                }
            )
            solved, outcome = self._solve_attempt(in_state, out_state, index, attempt=attempt + 1)
            if not solved:
                message = self._prepare_problems_message(outcome, index.db)
                exc = LibMambaUnsatisfiableError(message)
                exc.allow_retry = False
                raise exc

        # We didn't fail? Nice, let's return the calculated state
        self._export_solution(index, out_state, outcome)

        # Run post-solve tasks
        out_state.post_solve(solver=self)

        return out_state

    def _solve_attempt(
        self,
        in_state: SolverInputState,
        out_state: SolverOutputState,
        index: LibMambaIndexHelper,
        attempt: int = 1,
    ) -> tuple[bool, Solution | UnSolvable]:
        log.info("Solver attempt: #%d", attempt)
        log.debug("Current conflicts (including learnt ones): %s", out_state.conflicts)
        flags = self._solver_flags(in_state)
        jobs = self._specs_to_request_jobs(in_state, out_state)
        request = Request(jobs=jobs, flags=flags)
        solver = LibsolvSolver()
        outcome = solver.solve(index.db, request)
        if isinstance(outcome, Solution):
            out_state.conflicts.clear()
            return True, outcome
        old_conflicts = out_state.conflicts.copy()
        new_conflicts = self._maybe_raise_for_problems(outcome, index, old_conflicts)
        if log.isEnabledFor(logging.DEBUG):
            problems_as_str = outcome.problems_to_str(index.db)
            log.debug(
                "Attempt %d failed with %s conflicts:\n%s",
                attempt,
                len(new_conflicts),
                problems_as_str,
            )
        out_state.conflicts.update(new_conflicts)
        return False, outcome

    def _solver_flags(self, in_state: SolverInputState) -> Request.Flags:
        flags = Request.Flags()
        flags.allow_downgrade = True
        # About flags.allow_uninstall = True:
        # We used to set this to False on a global basis and then add jobs
        # individually with ALLOW_UNINSTALL=True. Libmamba v2 has a Keep job instead now.
        flags.allow_uninstall = True
        flags.force_reinstall = in_state.force_reinstall
        flags.keep_dependencies = True
        flags.keep_user_specs = True
        flags.order_request = False  # we do this ourselves
        flags.strict_repo_priority = context.channel_priority is ChannelPriority.STRICT
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
        for JobType, specs in jobs.items():
            for conda_spec in specs:
                libmamba_spec = self._conda_spec_to_libmamba_spec(conda_spec)
                request_jobs.append(JobType(libmamba_spec))
                if log.isEnabledFor(logging.INFO):
                    json_friendly.setdefault(JobType.__name__, []).append(str(conda_spec))
        if log.isEnabledFor(logging.INFO):
            json_str = json.dumps(json_friendly, indent=2)
            log.info("The solver will handle these requests:\n%s", json_str)
        return request_jobs

    def _specs_to_request_jobs_add(
        self,
        in_state: SolverInputState,
        out_state: SolverOutputState,
    ) -> dict[Request, list[MatchSpec | str]]:
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

        for name in out_state.specs:
            if name.startswith("__"):
                continue  # ignore virtual packages
            installed: PackageRecord = in_state.installed.get(name)
            if installed:
                installed_spec = self._check_spec_compat(installed.to_match_spec())
            else:
                installed_spec = None
            requested: MatchSpec = self._check_spec_compat(in_state.requested.get(name))
            history: MatchSpec = self._check_spec_compat(in_state.history.get(name))
            pinned: MatchSpec = self._check_spec_compat(in_state.pinned.get(name))
            conflicting: MatchSpec = self._check_spec_compat(out_state.conflicts.get(name))

            if name in user_installed and not in_state.prune and not conflicting:
                tasks[Request.Keep].append(installed_spec)

            # These specs are explicit in some sort of way
            if pinned and not pinned.is_name_only_spec:
                # these are the EXPLICIT pins; conda also uses implicit pinning to
                # constrain updates too but those can be overridden in case of conflicts.
                # name-only pins are treated as locks when installed, see below
                tasks[Request.Pin].append(pinned)
            # in libmamba, pins and installs are compatible tasks (pin only constrains,
            # does not 'request' a package). In classic, pins were actually targeted installs
            # so they were exclusive
            if requested:
                if requested.is_name_only_spec and pinned and not pinned.is_name_only_spec:
                    # for name-only specs, this is a no-op; we already added the pin above
                    # but we will constrain it again in the install task to have better
                    # error messages if not solvable
                    spec = pinned
                else:
                    spec = requested
                if installed:
                    tasks[Request.Update].append(spec)
                    tasks[Request.Keep].append(name)
                else:
                    tasks[Request.Install].append(spec)
            elif name in in_state.always_update:
                tasks[Request.Update].append(name)
                tasks[Request.Keep].append(name)
            # These specs are "implicit"; the solver logic massages them for better UX
            # as long as they don't cause trouble
            elif in_state.prune:
                continue
            elif name == "python" and installed and not pinned:
                pyver = ".".join(installed.version.split(".")[:2])
                tasks[Request.Pin].append(f"python {pyver}.*")
            elif history:
                if conflicting and history.strictness == 3:
                    # relax name-version-build (strictness=3) history specs that cause conflicts
                    # this is called neutering and makes test_neutering_of_historic_specs pass
                    spec_str = f"{name} {history.version}.*" if history.version else name
                    tasks[Request.Install].append(spec_str)
                else:
                    tasks[Request.Install].append(history)
            elif installed:
                if conflicting:
                    # NOTE: We don't do anything now with conflicting installed.
                    # We rely on Flags.allow_uninstall = True doing the right thing.
                    # We are protecting important things with Keep or Freeze instead.
                    pass
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
                        tasks[Request.Freeze].append(installed_spec)
                    else:
                        tasks[Request.Keep].append(name)

        # Sort tasks by priority
        # This ensures that more important tasks are added to the solver first
        returned_tasks = {}
        for task_type in (
            Request.Pin,
            Request.Install,
            Request.Update,
            Request.Keep,
            Request.Freeze,
        ):
            if task_type in tasks:
                returned_tasks[task_type] = tasks[task_type]
        return returned_tasks

    def _specs_to_request_jobs_remove(
        self, in_state: SolverInputState, out_state: SolverOutputState
    ) -> dict[Request, list[MatchSpec | str]]:
        # TODO: Consider merging add/remove in a single logic this so there's no split

        tasks = defaultdict(list)

        # Protect history and aggressive updates from being uninstalled if possible
        for name, record in out_state.records.items():
            if name in in_state.history or name in in_state.aggressive_updates:
                # MatchSpecs constructed from PackageRecords get parsed too
                # strictly if exported via str(). Use .conda_build_form() directly.
                spec = record.to_match_spec().conda_build_form()
                tasks[Request.Keep].append(spec)

        # No complications here: delete requested and their deps
        # TODO: There are some flags to take care of here, namely:
        # --all
        # --no-deps
        # --deps-only
        for name, spec in in_state.requested.items():
            spec = self._check_spec_compat(spec)
            tasks[Request.Remove].append(str(spec))

        return dict(tasks)

    def _specs_to_request_jobs_conda_build(
        self, in_state: SolverInputState, out_state: SolverOutputState
    ) -> dict[Request, list[MatchSpec | str]]:
        tasks = defaultdict(list)
        for name, spec in in_state.requested.items():
            if name.startswith("__"):
                continue
            spec = self._check_spec_compat(spec)
            spec = self._fix_version_field_for_conda_build(spec)
            tasks[Request.Install].append(spec.conda_build_form())

        return dict(tasks)

    def _export_solution(
        self,
        index: LibMambaIndexHelper,
        out_state: SolverOutputState,
        solution: Solution,
    ) -> SolverOutputState:
        for action in solution.actions:
            record_to_install: PackageInfo = getattr(action, "install", None)
            record_to_remove: PackageInfo = getattr(action, "remove", None)
            if record_to_install:
                if record_to_install.name.startswith("__"):
                    continue
                record = self._package_info_to_package_record(record_to_install, index)
                out_state.records[record.name] = record
            elif record_to_remove:
                if record_to_remove.name.startswith("__"):
                    continue
                record = self._package_info_to_package_record(record_to_remove, index)
                out_state.records.pop(record.name, None)
        return out_state

    def _package_info_to_package_record(
        self,
        pkg: PackageInfo,
        index: LibMambaIndexHelper,
    ) -> PackageRecord:
        if pkg.noarch == NoArchType.Python:
            noarch = "python"
        elif pkg.noarch == NoArchType.Generic:
            noarch = "generic"
        else:
            noarch = None
        # The package download logic needs the URL with credentials
        for repo_info in index.repos:
            if pkg.package_url.startswith(repo_info.url_no_cred):
                url = pkg.package_url.replace(repo_info.url_no_cred, repo_info.url_w_cred)
                break
        else:
            url = pkg.package_url
        return PackageRecord(
            name=pkg.name,
            version=pkg.version,
            build=pkg.build_string,  # NOTE: Different attribute name
            build_number=pkg.build_number,
            channel=pkg.channel,
            url=url,
            subdir=pkg.platform,  # NOTE: Different attribute name
            fn=pkg.filename,  # NOTE: Different attribute name
            license=pkg.license,
            md5=pkg.md5,
            sha256=pkg.sha256,
            signatures=pkg.signatures,
            track_features=pkg.track_features,
            depends=pkg.dependencies,  # NOTE: Different attribute name
            constrains=pkg.constrains,
            defaulted_keys=pkg.defaulted_keys,
            noarch=noarch,
            size=pkg.size,
            timestamp=pkg.timestamp,
        )

    # endregion

    # region Error reporting
    ########################

    @classmethod
    def _parse_problems(cls, unsolvable: UnSolvable, db: Database) -> Mapping[str, MatchSpec]:
        """
        Problems can signal either unsatisfiability or unavailability.
        First will raise LibmambaUnsatisfiableError.
        Second will raise PackagesNotFoundError.

        Libmamba can return spec strings in two formats:
        - With dashes, e.g. package-1.2.3-h5487548_0
        - à la conda-build, e.g. package 1.2.* *
        - just names, e.g. package
        """
        conflicts = []
        not_found = []
        problems = unsolvable.problems(db)
        try:
            explained_problems = unsolvable.explain_problems(db, _mamba_nocolor_palette)
        except Exception as exc:
            log.debug("Cannot explain problems", exc_info=exc)
            explained_problems = ""
        for line in problems:
            words = line.split()
            if "none of the providers can be installed" in line:
                if words[0] != "package" or words[2] != "requires":
                    raise ValueError(f"Unknown message: {line}")
                conflicts.append(cls._matchspec_from_error_str(words[1]))
                end = words.index("but")
                conflicts.append(cls._matchspec_from_error_str(words[3:end]))
            elif "- nothing provides" in line:
                marker = next((i for (i, w) in enumerate(words) if w == "needed"), None)
                if marker:
                    conflicts.append(cls._matchspec_from_error_str(words[-1]))
                start = 3 if marker == 4 else 4
                not_found.append(cls._matchspec_from_error_str(words[start:marker]))
            elif "has constraint" in line and "conflicting with" in line:
                # package libzlib-1.2.11-h4e544f5_1014 has constraint zlib 1.2.11 *_1014
                # conflicting with zlib-1.2.13-h998d150_0
                conflicts.append(cls._matchspec_from_error_str(words[-1]))
            elif "cannot install both pin-" in line and "and pin-" in line:
                # a pin is in conflict with another pin
                pin_a = words[3].rsplit("-", 1)[0]
                pin_b = words[5].rsplit("-", 1)[0]
                conflicts.append(MatchSpec(pin_a))
                conflicts.append(MatchSpec(pin_b))
            elif "is excluded by strict repo priority" in line:
                # package python-3.7.6-h0371630_2 is excluded by strict repo priority
                conflicts.append(cls._matchspec_from_error_str(words[1]))
            elif line == "unsupported request":
                # libmamba v2 has this message for package not found errors
                # we need to double check with the explained problem
                # if "does not exist" in explained_problems:
                for explained_line in explained_problems.splitlines():
                    explained_line = explained_line.strip()
                    explained_words = explained_line.split()
                    if "does not exist" in explained_line:
                        not_found.append(cls._matchspec_from_error_str(explained_words[1:3]))
                        break
            else:
                log.debug("! Problem line not recognized: %s", line)

        return {
            "conflicts": {s.name: s for s in conflicts},
            "not_found": {s.name: s for s in not_found},
        }

    def _maybe_raise_for_problems(
        self,
        unsolvable: UnSolvable,
        index: LibMambaIndexHelper,
        previous_conflicts: Mapping[str, MatchSpec] = None,
    ):
        parsed_problems = self._parse_problems(unsolvable, index.db)
        # We allow conda-build (if present) to process the exception early
        self._maybe_raise_for_conda_build(
            {**parsed_problems["conflicts"], **parsed_problems["not_found"]},
            message=self._prepare_problems_message(unsolvable, index.db),
        )

        unsatisfiable = parsed_problems["conflicts"]
        not_found = parsed_problems["not_found"]
        if not unsatisfiable and not_found:
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "Inferred PackagesNotFoundError %s from conflicts:\n%s",
                    tuple(not_found.keys()),
                    unsolvable.explain_problems(index.db, Palette.no_color()),
                )
            # This is not a conflict, but a missing package in the channel
            exc = PackagesNotFoundError(tuple(not_found.values()), tuple(index.channels))
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
            message = self._prepare_problems_message(unsolvable, index.db)
            exc = LibMambaUnsatisfiableError(message)
            # do not allow conda.cli.install to try more things
            exc.allow_retry = False
            raise exc
        return unsatisfiable

    def _prepare_problems_message(self, unsolvable: UnSolvable, db: Database) -> str:
        message = unsolvable.problems_to_str(db)
        explain = True
        if " - " not in message:
            # This makes 'explain_problems()' crash. Anticipate.
            message = "Failed with empty error message."
            explain = False
        elif "is excluded by strict repo priority" in message:
            # This will cause a lot of warnings until implemented in detail explanations
            log.info("Skipping error explanation. Excluded by strict repo priority.")
            explain = False

        if explain:
            try:
                explained_errors = unsolvable.explain_problems(db, _mamba_auto_palette)
                message += "\n" + explained_errors
            except Exception as exc:
                log.warning("Failed to explain problems", exc_info=exc)

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

    # endregion

    # region General helpers
    ########################

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
        if spec is None:
            return
        spec_fields = {}
        for field in spec.FIELD_NAMES:
            value = spec.get_raw_value(field)
            if value:
                if field == "channel" and str(value) == "<unknown>":
                    continue
                spec_fields[field] = value
        return MatchSpec(**spec_fields)

    def _conda_spec_to_libmamba_spec(self, spec: MatchSpec) -> LibmambaMatchSpec:
        return LibmambaMatchSpec.parse(str(spec))

    @staticmethod
    def _matchspec_from_error_str(spec: str | Sequence[str]):
        try:
            if isinstance(spec, str):
                name, version, build = spec.rsplit("-", 2)
                return MatchSpec(name=name, version=version, build=build)
            else:
                kwargs = {"name": spec[0].rstrip(",")}
                if len(spec) >= 2 and spec[1] != "=*":
                    if spec[1].startswith("==") or not spec[1].startswith("="):
                        kwargs["version"] = spec[1].rstrip(",")
                    else:
                        kwargs["version"] = spec[1][1:].rstrip(",") + ".*"
                if len(spec) == 3 and spec[2] != "*":
                    kwargs["build"] = spec[2].rstrip(",")
                return MatchSpec(**kwargs)
        except Exception as exc:
            raise ValueError(f"Could not parse spec: {spec}") from exc

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
        index_channels = {getattr(chn, "canonical_name", chn) for chn in index.channels}
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
