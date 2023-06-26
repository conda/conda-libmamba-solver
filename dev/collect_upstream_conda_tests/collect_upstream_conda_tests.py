"""
pytest plugin to modify which upstream (conda/conda) tests are run by pytest.
"""
from importlib.metadata import version

import pytest


_deselected_upstream_tests = [
    ### Deselect tests from conda/conda we cannot pass due to different reasons
    ### These used to be skipped or xfail'd upstream, but we are trying to
    ### keep it clean from this project's specifics
    # This test checks for plugin errors and assumes none are present, but
    # conda-libmamba-solver counts as one so we need to skip it.
    "tests/plugins/test_manager.py::test_load_entrypoints_importerror",
    # Conflict report / analysis is done differently with libmamba.
    "tests/cli/test_cli_install.py::test_find_conflicts_called_once",
    # SolverStateContainer needed
    "tests/core/test_solve.py::test_solve_2",
    "tests/core/test_solve.py::test_virtual_package_solver",
    "tests/core/test_solve.py::test_broken_install",
    # Features / nomkl involved
    "tests/core/test_solve.py::test_features_solve_1",
    "tests/core/test_solve.py::test_prune_1",
    "tests/test_create.py::test_remove_features",
    # Inconsistency analysis not implemented yet
    "tests/test_create.py::test_conda_recovery_of_pip_inconsistent_env",
    # Known bug in mamba; see https://github.com/mamba-org/mamba/issues/1197
    "tests/test_create.py::test_offline_with_empty_index_cache",
    # The following are known to fail upstream due to too strict expectations
    # We provide the same tests with adjusted checks in tests/test_modified_upstream.py
    "tests/test_create.py::test_neutering_of_historic_specs",
    "tests/test_create.py::test_pinned_override_with_explicit_spec",
    "tests/core/test_solve.py::test_pinned_1",
    "tests/core/test_solve.py::test_freeze_deps_1",
    "tests/core/test_solve.py::test_cuda_fail_1",
    "tests/core/test_solve.py::test_cuda_fail_2",
    "tests/core/test_solve.py::test_update_all_1",
    "tests/core/test_solve.py::test_conda_downgrade",
    "tests/core/test_solve.py::test_python2_update",
    "tests/core/test_solve.py::test_fast_update_with_update_modifier_not_set",
    "tests/core/test_solve.py::test_downgrade_python_prevented_with_sane_message",
    # These use libmamba-incompatible MatchSpecs (name[build_number=1] syntax)
    "tests/models/test_prefix_graph.py::test_deep_cyclical_dependency",
    # See https://github.com/conda/conda-libmamba-solver/pull/133#issuecomment-1448607110
    # These failed after enabling the whole unit test suite for `conda/conda`.
    # Errors are not critical but would require some further assessment in case fixes are obvious.
    "tests/cli/test_main_notices.py::test_notices_appear_once_when_running_decorated_commands",
    "tests/cli/test_main_notices.py::test_notices_does_not_interrupt_command_on_failure",
    "tests/conda_env/installers/test_pip.py::PipInstallerTest::test_stops_on_exception",
    "tests/conda_env/installers/test_pip.py::PipInstallerTest::test_straight_install",
    "tests/conda_env/specs/test_base.py::DetectTestCase::test_build_msg",
    "tests/conda_env/specs/test_base.py::DetectTestCase::test_dispatches_to_registered_specs",
    "tests/conda_env/specs/test_base.py::DetectTestCase::test_has_build_msg_function",
    "tests/conda_env/specs/test_base.py::DetectTestCase::test_passes_kwargs_to_all_specs",
    "tests/conda_env/specs/test_base.py::DetectTestCase::test_raises_exception_if_no_detection",
    # TODO: Fix upstream; they seem to assume no other solvers will be active via env var
    "tests/plugins/test_solvers.py::test_get_solver_backend",
    "tests/plugins/test_solvers.py::test_get_solver_backend_multiple",
    # TODO: Investigate these, since they are solver related-ish
    "tests/conda_env/specs/test_requirements.py::TestRequiremets::test_environment",
    "tests/models/test_prefix_graph.py::test_windows_sort_orders_1",
    # TODO: These ones need further investigation
    "tests/core/test_solve.py::test_channel_priority_churn_minimized",
    "tests/core/test_solve.py::test_priority_1",
    # TODO: Investigate why this fails on Windows now
    "tests/test_create.py::test_install_update_deps_only_deps_flags",
    # TODO: https://github.com/conda/conda-libmamba-solver/issues/141
    "tests/test_create.py::test_conda_pip_interop_conda_editable_package",
]

_broken_by_libmamba_1_4_2 = [
    # conda/tests
    "tests/core/test_solve.py::test_force_remove_1",
    "tests/core/test_solve.py::test_aggressive_update_packages",
    "tests/core/test_solve.py::test_update_deps_2",
    "tests/test_create.py::test_conda_pip_interop_dependency_satisfied_by_pip", # Linux-only
    "tests/test_create.py::test_conda_pip_interop_pip_clobbers_conda", # Linux-only
    "tests/test_create.py::test_install_tarball_from_local_channel", # Linux-only
    # conda-libmamba-solver/tests
    "tests/test_modified_upstream.py::test_pinned_1",
]


def pytest_collection_modifyitems(session, config, items):
    """
    We use this hook to modify which upstream tests (from the conda/conda repo)
    are run by pytest.

    This hook should not return anything but, instead, modify in place.
    """
    selected = []
    deselected = []
    for item in items:
        if item.nodeid in _deselected_upstream_tests:
            deselected.append(item)
            continue
        if version("libmambapy") >= "1.4.2" and item.nodeid in _broken_by_libmamba_1_4_2:
            item.add_marker(pytest.mark.xfail(reason="Broken by libmamba 1.4.2; see #186"))
        selected.append(item)
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)
