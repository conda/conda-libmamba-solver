import typing
import collections

import libmambapy as api

from conda_libmamba_solver.state import SolverInputState, SolverOutputState
from conda_libmamba_solver.solver import LibMambaSolver
from conda.models.match_spec import MatchSpec
from conda.base.constants import UpdateModifier
from conda.exceptions import CondaValueError


class MambaSolver(LibMambaSolver):
    def _specs_to_tasks_add(self, in_state: SolverInputState, out_state: SolverOutputState) -> typing.Dict:
        tasks = collections.defaultdict(list)

        # https://github.com/mamba-org/mamba/blob/main/mamba/mamba/mamba.py#L480
        if in_state.update_modifier == UpdateModifier.FREEZE_INSTALLED:
            tasks[('LOCK', api.SOLVER_LOCK)].extend([_.name for _ in out_state.records.values()])

        # https://github.com/mamba-org/mamba/blob/main/mamba/mamba/mamba.py#L376
        if self._command == "update" and in_state.update_modifier == UpdateModifier.UPDATE_ALL:
            tasks[("UPDATE", api.SOLVER_UPDATE)].extend(in_state.installed)

        # https://github.com/mamba-org/mamba/blob/main/mamba/mamba/mamba.py#L395
        elif self._command == "update" and in_state.update_modifier == UpdateModifier.UPDATE_DEPS:
            recursive_deps = list(in_state.requested.copy())
            for package_name in recursive_deps:
                if package_name not in in_state.installed:
                    raise CondaValueError(f'requested {package_name} not installed packages')
                for package_spec_str in out_state.records[package_name].depends:
                    ms = MatchSpec(package_spec_str)
                    if ms.name != "python":
                        # TODO: remove modifying the iterator (matching mamba approach)
                        recursive_deps.append(ms.name)
            tasks[("UPDATE", api.SOLVER_UPDATE)].extend(set(recursive_deps))

        # https://github.com/mamba-org/mamba/blob/main/mamba/mamba/mamba.py#L235
        # https://github.com/mamba-org/mamba/blob/main/mamba/mamba/mamba.py#L483
        requested_packages = [str(_) for _ in in_state.requested.values()]
        if self._command == "update":
            tasks[("UPDATE", api.SOLVER_UPDATE)].extend(requested_packages)
        else:
            tasks[("INSTALL", api.SOLVER_INSTALL)].extend(requested_packages)

        # https://github.com/mamba-org/mamba/blob/main/mamba/mamba/mamba.py#L485
        if not in_state.force_reinstall:
            for package_name in in_state.aggressive_updates:
                if package_name in in_state.installed:
                    tasks[('UPDATE', api.SOLVER_UPDATE)].append(package_name)

        # https://github.com/mamba-org/mamba/blob/main/mamba/mamba/mamba.py#L492
        if ("python" in in_state.installed and
            (
                in_state.update_modifier == UpdateModifier.UPDATE_ALL or
                "python" not in in_state.requested
             )
        ):
            python_version = in_state.installed["python"].version
            major_minor_version = ".".join(python_version.split(".")[:2])
            tasks[('ADD_PIN', api.SOLVER_NOOP)].append(f"python {major_minor_version}.*")

        # https://github.com/mamba-org/mamba/blob/main/mamba/mamba/mamba.py#L497
        for ms in in_state.pinned.values():
            tasks[('ADD_PIN', api.SOLVER_NOOP)].append(str(ms.conda_build_form()))

        return tasks

    def _specs_to_tasks_remove(self, in_state: SolverInputState, out_state: SolverOutputState) -> typing.Dict:
        tasks = collections.defaultdict(list)

        # https://github.com/mamba-org/mamba/blob/main/mamba/mamba/mamba.py#L183
        for spec in in_state.history.values():
            tasks[("USERINSTALLED", api.SOLVER_USERINSTALLED)].append(
                spec.conda_build_form())

        # https://github.com/mamba-org/mamba/blob/main/mamba/mamba/mamba.py#L190
        for spec in in_state.requested.values():
            tasks[('ERASE | CLEANDEPS', api.SOLVER_ERASE | api.SOLVER_CLEANDEPS)].append(
                spec.conda_build_form())

        return tasks
