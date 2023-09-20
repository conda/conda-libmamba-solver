# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
pytest plugin to modify which upstream (conda/conda) tests are run by pytest.
"""
from importlib.metadata import version

import pytest

# Deselect tests from conda/conda we cannot pass due to different reasons
# These used to be skipped or xfail'd upstream, but we are trying to
# keep it clean from this project's specifics
_deselected_upstream_tests = {
    # This test checks for plugin errors and assumes none are present, but
    # conda-libmamba-solver counts as one so we need to skip it.
    "tests/plugins/test_manager.py": ["test_load_entrypoints_importerror"],
    # Conflict report / analysis is done differently with libmamba.
    "tests/cli/test_cli_install.py": ["test_find_conflicts_called_once"],
    "tests/core/test_solve.py": [
        # SolverStateContainer needed
        "test_solve_2",
        "test_virtual_package_solver",
        "test_broken_install",
        # Features / nomkl involved
        "test_features_solve_1",
        "test_prune_1",
        "test_update_prune_2",
        "test_update_prune_3",
        # Message expected, but libmamba does not report constraints
        "test_update_prune_5",
        # classic expects implicit update to channel with higher priority, including downgrades
        # libmamba does not do this, it just stays in the same channel; should it change?
        "test_priority_1",
        # The following are known to fail upstream due to too strict expectations
        # We provide the same tests with adjusted checks in tests/test_modified_upstream.py
        "test_pinned_1",
        "test_freeze_deps_1",
        "test_cuda_fail_1",
        "test_cuda_fail_2",
        "test_update_all_1",
        "test_conda_downgrade",
        "test_python2_update",
        "test_fast_update_with_update_modifier_not_set",
        "test_downgrade_python_prevented_with_sane_message",
    ],
    "tests/test_create.py": [
        # libmamba does not support features
        "test_remove_features",
        # Known bug in mamba; see https://github.com/mamba-org/mamba/issues/1197
        "test_offline_with_empty_index_cache",
        # Adjusted in tests/test_modified_upstream.py
        "test_install_features",
        "test_pinned_override_with_explicit_spec",
        # TODO: https://github.com/conda/conda-libmamba-solver/issues/141
        "test_conda_pip_interop_conda_editable_package",
    ],
    # These use libmamba-incompatible MatchSpecs (name[build_number=1] syntax)
    "tests/models/test_prefix_graph.py": [
        "test_deep_cyclical_dependency",
        # TODO: Investigate this, since they are solver related-ish
        "test_windows_sort_orders_1",
    ],
    # See https://github.com/conda/conda-libmamba-solver/pull/133#issuecomment-1448607110
    # These failed after enabling the whole unit test suite for `conda/conda`.
    # Errors are not critical but would require some further assessment in case fixes are obvious.
    "tests/cli/test_main_notices.py": [
        "test_notices_appear_once_when_running_decorated_commands",
        "test_notices_does_not_interrupt_command_on_failure",
    ],
    "tests/conda_env/installers/test_pip.py": [
        "PipInstallerTest::test_stops_on_exception",
        "PipInstallerTest::test_straight_install",
    ],
    "tests/conda_env/specs/test_base.py": [
        "DetectTestCase::test_build_msg",
        "DetectTestCase::test_dispatches_to_registered_specs",
        "DetectTestCase::test_has_build_msg_function",
        "DetectTestCase::test_passes_kwargs_to_all_specs",
        "DetectTestCase::test_raises_exception_if_no_detection",
    ],
    # TODO: Fix upstream; they seem to assume no other solvers will be active via env var
    "tests/plugins/test_solvers.py": [
        "test_get_solver_backend",
        "test_get_solver_backend_multiple",
    ],
    # TODO: Investigate these, since they are solver related-ish
    "tests/conda_env/specs/test_requirements.py": [
        "TestRequirements::test_environment",
    ],
    # Added to test_modified_upstream.py
    "tests/test_priority.py": ["test_reorder_channel_priority"],
    # Added to test_modified_upstream.py; this passes just by moving it to another test file
    "tests/test_misc.py": ["test_explicit_missing_cache_entries"],
}


_broken_by_libmamba_1_5_x = {
    # conda/tests
    "tests/test_export.py": [
        "test_explicit",
        "test_export",
    ],
}


def pytest_collection_modifyitems(session, config, items):
    """
    We use this hook to modify which upstream tests (from the conda/conda repo)
    are run by pytest.

    This hook should not return anything but, instead, modify in place.
    """
    selected = []
    deselected = []
    for item in items:
        path_key = "/".join(item.path.parts[item.path.parts.index("tests") :])
        item_name_no_brackets = item.name.split("[")[0]
        if item_name_no_brackets in _deselected_upstream_tests.get(path_key, []):
            deselected.append(item)
            continue
        if version("libmambapy").startswith(
            "1.5."
        ) and item_name_no_brackets in _broken_by_libmamba_1_5_x.get(path_key, []):
            item.add_marker(
                pytest.mark.xfail(
                    reason="Broken in libmamba 1.5.x; "
                    "see https://github.com/mamba-org/mamba/issues/2431."
                )
            )
        selected.append(item)
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)
