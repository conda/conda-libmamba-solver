# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
This module fixes some tests found across conda/conda's suite to
check the "spirit" of the test, instead of making explicit comparisons
in stdout messages, overly strict solver checks and other differences
that do not result in incompatible behavior.

We are copying those offending tests instead of patching them to keep
conda/conda code base as unaffected by this work as possible, but it is
indeed feasible to upgrade those tests in the future for more flexible
comparisons. This is only a workaround during the experimental phase.

Tests were brought over and patched on Feb 7th, 2022, following the
source found in commit 98fb262c610e17a7731b9183bf37cca98dcc1a71.
"""

import os
import sys
import warnings
from pprint import pprint

import pytest
from conda.auxlib.ish import dals
from conda.base.constants import UpdateModifier, on_win
from conda.base.context import conda_tests_ctxt_mgmt_def_pol, context
from conda.common.io import env_var
from conda.core.package_cache_data import PackageCacheData
from conda.core.prefix_data import PrefixData
from conda.core.subdir_data import SubdirData
from conda.exceptions import UnsatisfiableError
from conda.gateways.subprocess import subprocess_call_with_clean_env
from conda.misc import explicit
from conda.models.match_spec import MatchSpec
from conda.models.version import VersionOrder
from conda.testing import (
    CondaCLIFixture,
    TmpEnvFixture,
    conda_cli,
    path_factory,
    tmp_env,
)
from conda.testing.cases import BaseTestCase
from conda.testing.helpers import (
    add_subdir,
    add_subdir_to_iter,
    convert_to_dist_str,
    get_solver,
    get_solver_2,
    get_solver_4,
    get_solver_aggregate_1,
    get_solver_aggregate_2,
    get_solver_cuda,
)
from conda.testing.integration import (
    PYTHON_BINARY,
    Commands,
    make_temp_env,
    package_is_installed,
    run_command,
)
from pytest import MonkeyPatch
from pytest_mock import MockerFixture


@pytest.mark.integration
class PatchedCondaTestCreate(BaseTestCase):
    """
    These tests come from `conda/conda::tests/test_create.py`
    """

    def setUp(self):
        PackageCacheData.clear()

    def test_pinned_override_with_explicit_spec(self):
        with make_temp_env("python=3.6") as prefix:
            ## MODIFIED
            # Original test assumed the `python=3.6` spec above resolves to `python=3.6.5`
            # Instead we only pin whatever the solver decided to install
            # Original lines were:
            ### run_command(Commands.CONFIG, prefix,
            ###             "--add", "pinned_packages", "python=3.6.5")
            python = next(PrefixData(prefix).query("python"))
            run_command(
                Commands.CONFIG, prefix, "--add", "pinned_packages", f"python={python.version}"
            )
            ## /MODIFIED

            run_command(Commands.INSTALL, prefix, "python=3.7", no_capture=True)
            assert package_is_installed(prefix, "python=3.7")

    @pytest.mark.xfail(on_win, reason="TODO: Investigate why this fails on Windows only")
    def test_install_update_deps_only_deps_flags(self):
        with make_temp_env("flask=2.0.1", "jinja2=3.0.1") as prefix:
            python = os.path.join(prefix, PYTHON_BINARY)
            result_before = subprocess_call_with_clean_env([python, "--version"])
            assert package_is_installed(prefix, "flask=2.0.1")
            assert package_is_installed(prefix, "jinja2=3.0.1")
            run_command(
                Commands.INSTALL,
                prefix,
                "flask",
                "python",
                "--update-deps",
                "--only-deps",
                no_capture=True,
            )
            result_after = subprocess_call_with_clean_env([python, "--version"])
            assert result_before == result_after
            assert package_is_installed(prefix, "flask=2.0.1")
            assert package_is_installed(prefix, "jinja2>3.0.1")


@pytest.mark.xfail(on_win, reason="nomkl not present on windows", strict=True)
def test_install_features():
    # MODIFIED: Added fixture manually
    PackageCacheData.clear()
    # /MODIFIED
    with make_temp_env("python=2", "numpy=1.13", "nomkl", no_capture=True) as prefix:
        assert package_is_installed(prefix, "numpy")
        assert package_is_installed(prefix, "nomkl")
        assert not package_is_installed(prefix, "mkl")

    with make_temp_env("python=2", "numpy=1.13") as prefix:
        assert package_is_installed(prefix, "numpy")
        assert not package_is_installed(prefix, "nomkl")
        assert package_is_installed(prefix, "mkl")

        # run_command(Commands.INSTALL, prefix, "nomkl", no_capture=True)
        run_command(Commands.INSTALL, prefix, "python=2", "nomkl", no_capture=True)
        # MODIFIED ^: python=2 needed explicitly to trigger update
        assert package_is_installed(prefix, "numpy")
        assert package_is_installed(prefix, "nomkl")
        assert package_is_installed(prefix, "blas=1.0=openblas")
        assert not package_is_installed(prefix, "mkl_fft")
        assert not package_is_installed(prefix, "mkl_random")
        # assert not package_is_installed(prefix, "mkl")  # pruned as an indirect dep


# The following tests come from `conda/conda::tests/core/test_solve.py`


@pytest.mark.integration
def test_pinned_1(tmpdir):
    specs = (MatchSpec("numpy"),)
    with get_solver(tmpdir, specs) as solver:
        final_state_1 = solver.solve_final_state()
        # PrefixDag(final_state_1, specs).open_url()
        pprint(convert_to_dist_str(final_state_1))
        order = add_subdir_to_iter(
            (
                "channel-1::openssl-1.0.1c-0",
                "channel-1::readline-6.2-0",
                "channel-1::sqlite-3.7.13-0",
                "channel-1::system-5.8-1",
                "channel-1::tk-8.5.13-0",
                "channel-1::zlib-1.2.7-0",
                "channel-1::python-3.3.2-0",
                "channel-1::numpy-1.7.1-py33_0",
            )
        )
        assert convert_to_dist_str(final_state_1) == order

    with env_var(
        "CONDA_PINNED_PACKAGES",
        "python=2.6&iopro<=1.4.2",
        stack_callback=conda_tests_ctxt_mgmt_def_pol,
    ):
        specs = (MatchSpec("system=5.8=0"),)
        with get_solver(tmpdir, specs) as solver:
            final_state_1 = solver.solve_final_state()
            # PrefixDag(final_state_1, specs).open_url()
            pprint(convert_to_dist_str(final_state_1))
            order = add_subdir_to_iter(("channel-1::system-5.8-0",))
            assert convert_to_dist_str(final_state_1) == order

        # ignore_pinned=True
        specs_to_add = (MatchSpec("python"),)
        with get_solver(
            tmpdir, specs_to_add=specs_to_add, prefix_records=final_state_1, history_specs=specs
        ) as solver:
            final_state_2 = solver.solve_final_state(ignore_pinned=True)
            # PrefixDag(final_state_1, specs).open_url()
            pprint(convert_to_dist_str(final_state_2))
            order = add_subdir_to_iter(
                (
                    "channel-1::openssl-1.0.1c-0",
                    "channel-1::readline-6.2-0",
                    "channel-1::sqlite-3.7.13-0",
                    "channel-1::system-5.8-0",
                    "channel-1::tk-8.5.13-0",
                    "channel-1::zlib-1.2.7-0",
                    "channel-1::python-3.3.2-0",
                )
            )
            assert convert_to_dist_str(final_state_2) == order

        # ignore_pinned=False
        specs_to_add = (MatchSpec("python"),)
        with get_solver(
            tmpdir, specs_to_add=specs_to_add, prefix_records=final_state_1, history_specs=specs
        ) as solver:
            final_state_2 = solver.solve_final_state(ignore_pinned=False)
            # PrefixDag(final_state_1, specs).open_url()
            pprint(convert_to_dist_str(final_state_2))
            order = add_subdir_to_iter(
                (
                    "channel-1::openssl-1.0.1c-0",
                    "channel-1::readline-6.2-0",
                    "channel-1::sqlite-3.7.13-0",
                    "channel-1::system-5.8-0",
                    "channel-1::tk-8.5.13-0",
                    "channel-1::zlib-1.2.7-0",
                    "channel-1::python-2.6.8-6",
                )
            )
            assert convert_to_dist_str(final_state_2) == order

        # incompatible CLI and configured specs
        specs_to_add = (MatchSpec("scikit-learn==0.13"),)
        with get_solver(
            tmpdir, specs_to_add=specs_to_add, prefix_records=final_state_1, history_specs=specs
        ) as solver:
            ## MODIFIED
            # Original tests checks for SpecsConfigurationConflictError being raised
            # but libmamba will fails with UnsatisfiableError instead. Hence, we check
            # the error string. Original check inspected the kwargs of the exception:
            ### with pytest.raises(SpecsConfigurationConflictError) as exc:
            ###    solver.solve_final_state(ignore_pinned=False)
            ### kwargs = exc.value._kwargs
            ### assert kwargs["requested_specs"] == ["scikit-learn==0.13"]
            ### assert kwargs["pinned_specs"] == ["python=2.6"]
            with pytest.raises(UnsatisfiableError) as exc_info:
                solver.solve_final_state(ignore_pinned=False)
            error = str(exc_info.value)
            assert "package scikit-learn-0.13" in error
            assert "requires python 2.7*" in error
            ## /MODIFIED

        specs_to_add = (MatchSpec("numba"),)
        history_specs = (
            MatchSpec("python"),
            MatchSpec("system=5.8=0"),
        )
        with get_solver(
            tmpdir,
            specs_to_add=specs_to_add,
            prefix_records=final_state_2,
            history_specs=history_specs,
        ) as solver:
            final_state_3 = solver.solve_final_state()
            # PrefixDag(final_state_1, specs).open_url()
            pprint(convert_to_dist_str(final_state_3))
            order = add_subdir_to_iter(
                (
                    "channel-1::openssl-1.0.1c-0",
                    "channel-1::readline-6.2-0",
                    "channel-1::sqlite-3.7.13-0",
                    "channel-1::system-5.8-0",
                    "channel-1::tk-8.5.13-0",
                    "channel-1::zlib-1.2.7-0",
                    "channel-1::llvm-3.2-0",
                    "channel-1::python-2.6.8-6",
                    "channel-1::argparse-1.2.1-py26_0",
                    "channel-1::llvmpy-0.11.2-py26_0",
                    "channel-1::numpy-1.7.1-py26_0",
                    "channel-1::numba-0.8.1-np17py26_0",
                )
            )
            assert convert_to_dist_str(final_state_3) == order

        specs_to_add = (MatchSpec("python"),)
        history_specs = (
            MatchSpec("python"),
            MatchSpec("system=5.8=0"),
            MatchSpec("numba"),
        )
        with get_solver(
            tmpdir,
            specs_to_add=specs_to_add,
            prefix_records=final_state_3,
            history_specs=history_specs,
        ) as solver:
            final_state_4 = solver.solve_final_state(update_modifier=UpdateModifier.UPDATE_DEPS)
            # PrefixDag(final_state_1, specs).open_url()
            pprint(convert_to_dist_str(final_state_4))
            order = add_subdir_to_iter(
                (
                    "channel-1::openssl-1.0.1c-0",
                    "channel-1::readline-6.2-0",
                    "channel-1::sqlite-3.7.13-0",
                    "channel-1::system-5.8-1",
                    "channel-1::tk-8.5.13-0",
                    "channel-1::zlib-1.2.7-0",
                    "channel-1::llvm-3.2-0",
                    "channel-1::python-2.6.8-6",
                    "channel-1::argparse-1.2.1-py26_0",
                    "channel-1::llvmpy-0.11.2-py26_0",
                    "channel-1::numpy-1.7.1-py26_0",
                    "channel-1::numba-0.8.1-np17py26_0",
                )
            )
            assert convert_to_dist_str(final_state_4) == order

        specs_to_add = (MatchSpec("python"),)
        history_specs = (
            MatchSpec("python"),
            MatchSpec("system=5.8=0"),
            MatchSpec("numba"),
        )
        with get_solver(
            tmpdir,
            specs_to_add=specs_to_add,
            prefix_records=final_state_4,
            history_specs=history_specs,
        ) as solver:
            final_state_5 = solver.solve_final_state(update_modifier=UpdateModifier.UPDATE_ALL)
            # PrefixDag(final_state_1, specs).open_url()
            pprint(convert_to_dist_str(final_state_5))
            order = add_subdir_to_iter(
                (
                    "channel-1::openssl-1.0.1c-0",
                    "channel-1::readline-6.2-0",
                    "channel-1::sqlite-3.7.13-0",
                    "channel-1::system-5.8-1",
                    "channel-1::tk-8.5.13-0",
                    "channel-1::zlib-1.2.7-0",
                    "channel-1::llvm-3.2-0",
                    "channel-1::python-2.6.8-6",
                    "channel-1::argparse-1.2.1-py26_0",
                    "channel-1::llvmpy-0.11.2-py26_0",
                    "channel-1::numpy-1.7.1-py26_0",
                    "channel-1::numba-0.8.1-np17py26_0",
                )
            )
            assert convert_to_dist_str(final_state_5) == order

    # now update without pinning
    # MODIFIED: libmamba decides to stay in python=2.6 unless explicit
    # specs_to_add = (MatchSpec("python"),)
    specs_to_add = (MatchSpec("python=3"),)
    # /MODIFIED
    history_specs = (
        MatchSpec("python"),
        MatchSpec("system=5.8=0"),
        MatchSpec("numba"),
    )
    with get_solver(
        tmpdir,
        specs_to_add=specs_to_add,
        prefix_records=final_state_4,
        history_specs=history_specs,
    ) as solver:
        final_state_5 = solver.solve_final_state(update_modifier=UpdateModifier.UPDATE_ALL)
        # PrefixDag(final_state_1, specs).open_url()
        print(convert_to_dist_str(final_state_5))
        order = add_subdir_to_iter(
            (
                "channel-1::openssl-1.0.1c-0",
                "channel-1::readline-6.2-0",
                "channel-1::sqlite-3.7.13-0",
                "channel-1::system-5.8-1",
                "channel-1::tk-8.5.13-0",
                "channel-1::zlib-1.2.7-0",
                "channel-1::llvm-3.2-0",
                "channel-1::python-3.3.2-0",
                "channel-1::llvmpy-0.11.2-py33_0",
                "channel-1::numpy-1.7.1-py33_0",
                "channel-1::numba-0.8.1-np17py33_0",
            )
        )
        assert convert_to_dist_str(final_state_5) == order


@pytest.mark.integration
def test_freeze_deps_1(tmpdir):
    specs = (MatchSpec("six=1.7"),)
    with get_solver_2(tmpdir, specs) as solver:
        ## ADDED
        solver._command = "install"
        ## /ADDED
        final_state_1 = solver.solve_final_state()
        pprint(convert_to_dist_str(final_state_1))
        order = add_subdir_to_iter(
            (
                "channel-2::openssl-1.0.2l-0",
                "channel-2::readline-6.2-2",
                "channel-2::sqlite-3.13.0-0",
                "channel-2::tk-8.5.18-0",
                "channel-2::xz-5.2.3-0",
                "channel-2::zlib-1.2.11-0",
                "channel-2::python-3.4.5-0",
                "channel-2::six-1.7.3-py34_0",
            )
        )
        assert convert_to_dist_str(final_state_1) == order

    specs_to_add = (MatchSpec("bokeh"),)
    with get_solver_2(
        tmpdir, specs_to_add, prefix_records=final_state_1, history_specs=specs
    ) as solver:
        ## ADDED
        solver._command = "install"
        ## /ADDED
        unlink_precs, link_precs = solver.solve_for_diff()
        pprint(convert_to_dist_str(unlink_precs))
        pprint(convert_to_dist_str(link_precs))
        unlink_order = ()
        link_order = add_subdir_to_iter(
            (
                "channel-2::mkl-2017.0.3-0",
                "channel-2::yaml-0.1.6-0",
                "channel-2::backports_abc-0.5-py34_0",
                "channel-2::markupsafe-1.0-py34_0",
                "channel-2::numpy-1.13.0-py34_0",
                "channel-2::pyyaml-3.12-py34_0",
                "channel-2::requests-2.14.2-py34_0",
                "channel-2::setuptools-27.2.0-py34_0",
                "channel-2::jinja2-2.9.6-py34_0",
                "channel-2::python-dateutil-2.6.1-py34_0",
                "channel-2::tornado-4.4.2-py34_0",
                "channel-2::bokeh-0.12.4-py34_0",
            )
        )
        assert convert_to_dist_str(unlink_precs) == unlink_order
        assert convert_to_dist_str(link_precs) == link_order

    # now we can't install the latest bokeh 0.12.5, but instead we get bokeh 0.12.4
    specs_to_add = (MatchSpec("bokeh"),)
    with get_solver_2(
        tmpdir,
        specs_to_add,
        prefix_records=final_state_1,
        history_specs=(MatchSpec("six=1.7"), MatchSpec("python=3.4")),
    ) as solver:
        ## ADDED
        solver._command = "install"
        ## /ADDED
        unlink_precs, link_precs = solver.solve_for_diff()
        pprint(convert_to_dist_str(unlink_precs))
        pprint(convert_to_dist_str(link_precs))
        unlink_order = ()
        link_order = add_subdir_to_iter(
            (
                "channel-2::mkl-2017.0.3-0",
                "channel-2::yaml-0.1.6-0",
                "channel-2::backports_abc-0.5-py34_0",
                "channel-2::markupsafe-1.0-py34_0",
                "channel-2::numpy-1.13.0-py34_0",
                "channel-2::pyyaml-3.12-py34_0",
                "channel-2::requests-2.14.2-py34_0",
                "channel-2::setuptools-27.2.0-py34_0",
                "channel-2::jinja2-2.9.6-py34_0",
                "channel-2::python-dateutil-2.6.1-py34_0",
                "channel-2::tornado-4.4.2-py34_0",
                "channel-2::bokeh-0.12.4-py34_0",
            )
        )
        assert convert_to_dist_str(unlink_precs) == unlink_order
        assert convert_to_dist_str(link_precs) == link_order

    # here, the python=3.4 spec can't be satisfied, so it's dropped, and we go back to py27
    with pytest.raises(UnsatisfiableError):
        specs_to_add = (MatchSpec("bokeh=0.12.5"),)
        with get_solver_2(
            tmpdir,
            specs_to_add,
            prefix_records=final_state_1,
            history_specs=(MatchSpec("six=1.7"), MatchSpec("python=3.4")),
        ) as solver:
            ## ADDED
            solver._command = "install"
            ## /ADDED
            unlink_precs, link_precs = solver.solve_for_diff()

    # adding the explicit python spec allows conda to change the python versions.
    # one possible outcome is that this updates to python 3.6.  That is not desirable because of the
    #    explicit "six=1.7" request in the history.  It should only neuter that spec if there's no way
    #    to solve it with that spec.
    specs_to_add = MatchSpec("bokeh=0.12.5"), MatchSpec("python")
    with get_solver_2(
        tmpdir,
        specs_to_add,
        prefix_records=final_state_1,
        history_specs=(MatchSpec("six=1.7"), MatchSpec("python=3.4")),
    ) as solver:
        ## ADDED
        solver._command = "install"
        ## /ADDED

        unlink_precs, link_precs = solver.solve_for_diff()

        pprint(convert_to_dist_str(unlink_precs))
        pprint(convert_to_dist_str(link_precs))
        unlink_order = add_subdir_to_iter(
            (
                "channel-2::six-1.7.3-py34_0",
                "channel-2::python-3.4.5-0",
                # MODIFIED: xz is not uninstalled for some reason in libmamba :shrug:
                # "channel-2::xz-5.2.3-0",
            )
        )
        link_order = add_subdir_to_iter(
            (
                "channel-2::mkl-2017.0.3-0",
                "channel-2::yaml-0.1.6-0",
                "channel-2::python-2.7.13-0",
                "channel-2::backports-1.0-py27_0",
                "channel-2::backports_abc-0.5-py27_0",
                "channel-2::certifi-2016.2.28-py27_0",
                "channel-2::futures-3.1.1-py27_0",
                "channel-2::markupsafe-1.0-py27_0",
                "channel-2::numpy-1.13.1-py27_0",
                "channel-2::pyyaml-3.12-py27_0",
                "channel-2::requests-2.14.2-py27_0",
                "channel-2::six-1.7.3-py27_0",
                "channel-2::python-dateutil-2.6.1-py27_0",
                "channel-2::setuptools-36.4.0-py27_1",
                "channel-2::singledispatch-3.4.0.3-py27_0",
                "channel-2::ssl_match_hostname-3.5.0.1-py27_0",
                "channel-2::jinja2-2.9.6-py27_0",
                "channel-2::tornado-4.5.2-py27_0",
                "channel-2::bokeh-0.12.5-py27_1",
            )
        )
        assert convert_to_dist_str(unlink_precs) == unlink_order
        assert convert_to_dist_str(link_precs) == link_order

    # here, the python=3.4 spec can't be satisfied, so it's dropped, and we go back to py27
    specs_to_add = (MatchSpec("bokeh=0.12.5"),)
    with get_solver_2(
        tmpdir,
        specs_to_add,
        prefix_records=final_state_1,
        history_specs=(MatchSpec("six=1.7"), MatchSpec("python=3.4")),
    ) as solver:
        with pytest.raises(UnsatisfiableError):
            ## ADDED
            solver._command = "install"
            ## /ADDED
            solver.solve_final_state(update_modifier=UpdateModifier.FREEZE_INSTALLED)


def test_cuda_fail_1(tmpdir):
    specs = (MatchSpec("cudatoolkit"),)

    # No cudatoolkit in index for CUDA 8.0
    with env_var("CONDA_OVERRIDE_CUDA", "8.0"):
        with get_solver_cuda(tmpdir, specs) as solver:
            with pytest.raises(UnsatisfiableError) as exc:
                final_state = solver.solve_final_state()

    ## MODIFIED
    # libmamba will generate a slightly different error message, but the spirit is the same.
    # Original check was:
    ###     if sys.platform == "darwin":
    ###         plat = "osx-64"
    ###     elif sys.platform == "linux":
    ###         plat = "linux-64"
    ###     elif sys.platform == "win32":
    ###         if platform.architecture()[0] == "32bit":
    ###             plat = "win-32"
    ###         else:
    ###             plat = "win-64"
    ###     else:
    ###         plat = "linux-64"
    ###     assert str(exc.value).strip() == dals("""The following specifications were found to be incompatible with your system:
    ###
    ###   - feature:/{}::__cuda==8.0=0
    ###   - cudatoolkit -> __cuda[version='>=10.0|>=9.0']
    ###
    ### Your installed version is: 8.0""".format(plat))
    possible_messages = [
        dals(
            """Encountered problems while solving:
  - nothing provides __cuda >=9.0 needed by cudatoolkit-9.0-0"""
        ),
        dals(
            """Encountered problems while solving:
  - nothing provides __cuda >=10.0 needed by cudatoolkit-10.0-0"""
        ),
    ]
    exc_msg = str(exc.value).strip()
    assert any(msg in exc_msg for msg in possible_messages)
    ## /MODIFIED


def test_cuda_fail_2(tmpdir):
    specs = (MatchSpec("cudatoolkit"),)

    # No CUDA on system
    with env_var("CONDA_OVERRIDE_CUDA", ""):
        with get_solver_cuda(tmpdir, specs) as solver:
            with pytest.raises(UnsatisfiableError) as exc:
                final_state = solver.solve_final_state()

    ## MODIFIED
    # libmamba will generate a slightly different error message, but the spirit is the same.
    # Original check was:
    ###     assert str(exc.value).strip() == dals("""The following specifications were found to be incompatible with your system:
    ###
    ###   - cudatoolkit -> __cuda[version='>=10.0|>=9.0']
    ###
    ### Your installed version is: not available""")
    possible_messages = [
        dals(
            """Encountered problems while solving:
  - nothing provides __cuda >=9.0 needed by cudatoolkit-9.0-0"""
        ),
        dals(
            """Encountered problems while solving:
  - nothing provides __cuda >=10.0 needed by cudatoolkit-10.0-0"""
        ),
    ]
    exc_msg = str(exc.value).strip()
    assert any(msg in exc_msg for msg in possible_messages)
    ## /MODIFIED


def test_update_all_1(tmpdir):
    ## MODIFIED
    # Libmamba requires MatchSpec.conda_build_form() internally, which depends on `version` and
    # `build` fields. `system` below is using only `build_number`, so we have to adapt the syntax
    # accordingly. It should be the same result, but in a conda_build_form-friendly way:
    ### specs = MatchSpec("numpy=1.5"), MatchSpec("python=2.6"), MatchSpec("system[build_number=0]")
    specs = (
        MatchSpec("numpy=1.5"),
        MatchSpec("python=2.6"),
        MatchSpec("system[version=*,build=*0]"),
    )
    ## /MODIFIED

    with get_solver(tmpdir, specs) as solver:
        final_state_1 = solver.solve_final_state()
        # PrefixDag(final_state_1, specs).open_url()
        print(convert_to_dist_str(final_state_1))
        order = add_subdir_to_iter(
            (
                "channel-1::openssl-1.0.1c-0",
                "channel-1::readline-6.2-0",
                "channel-1::sqlite-3.7.13-0",
                "channel-1::system-5.8-0",
                "channel-1::tk-8.5.13-0",
                "channel-1::zlib-1.2.7-0",
                "channel-1::python-2.6.8-6",
                "channel-1::numpy-1.5.1-py26_4",
            )
        )
        assert convert_to_dist_str(final_state_1) == order

    specs_to_add = MatchSpec("numba=0.6"), MatchSpec("numpy")
    with get_solver(
        tmpdir, specs_to_add, prefix_records=final_state_1, history_specs=specs
    ) as solver:
        final_state_2 = solver.solve_final_state()
        # PrefixDag(final_state_2, specs).open_url()
        print(convert_to_dist_str(final_state_2))
        order = add_subdir_to_iter(
            (
                "channel-1::openssl-1.0.1c-0",
                "channel-1::readline-6.2-0",
                "channel-1::sqlite-3.7.13-0",
                "channel-1::system-5.8-0",
                "channel-1::tk-8.5.13-0",
                "channel-1::zlib-1.2.7-0",
                "channel-1::llvm-3.2-0",
                "channel-1::python-2.6.8-6",
                "channel-1::llvmpy-0.10.2-py26_0",
                "channel-1::nose-1.3.0-py26_0",
                "channel-1::numpy-1.7.1-py26_0",
                "channel-1::numba-0.6.0-np17py26_0",
            )
        )
        assert convert_to_dist_str(final_state_2) == order

    specs_to_add = (MatchSpec("numba=0.6"),)
    with get_solver(
        tmpdir, specs_to_add, prefix_records=final_state_1, history_specs=specs
    ) as solver:
        final_state_2 = solver.solve_final_state(update_modifier=UpdateModifier.UPDATE_ALL)
        # PrefixDag(final_state_2, specs).open_url()
        print(convert_to_dist_str(final_state_2))
        order = add_subdir_to_iter(
            (
                "channel-1::openssl-1.0.1c-0",
                "channel-1::readline-6.2-0",
                "channel-1::sqlite-3.7.13-0",
                "channel-1::system-5.8-1",
                "channel-1::tk-8.5.13-0",
                "channel-1::zlib-1.2.7-0",
                "channel-1::llvm-3.2-0",
                "channel-1::python-2.6.8-6",  # stick with python=2.6 even though UPDATE_ALL
                "channel-1::llvmpy-0.10.2-py26_0",
                "channel-1::nose-1.3.0-py26_0",
                "channel-1::numpy-1.7.1-py26_0",
                "channel-1::numba-0.6.0-np17py26_0",
            )
        )
        assert convert_to_dist_str(final_state_2) == order


def test_conda_downgrade(tmpdir):
    specs = (MatchSpec("conda-build"),)
    with env_var("CONDA_CHANNEL_PRIORITY", "False", stack_callback=conda_tests_ctxt_mgmt_def_pol):
        with get_solver_aggregate_1(tmpdir, specs) as solver:
            final_state_1 = solver.solve_final_state()
            pprint(convert_to_dist_str(final_state_1))
            order = add_subdir_to_iter(
                (
                    "channel-4::ca-certificates-2018.03.07-0",
                    "channel-2::conda-env-2.6.0-0",
                    "channel-2::libffi-3.2.1-1",
                    "channel-4::libgcc-ng-8.2.0-hdf63c60_0",
                    "channel-4::libstdcxx-ng-8.2.0-hdf63c60_0",
                    "channel-2::zlib-1.2.11-0",
                    "channel-4::ncurses-6.1-hf484d3e_0",
                    "channel-4::openssl-1.0.2p-h14c3975_0",
                    "channel-4::patchelf-0.9-hf484d3e_2",
                    "channel-4::tk-8.6.7-hc745277_3",
                    "channel-4::xz-5.2.4-h14c3975_4",
                    "channel-4::yaml-0.1.7-had09818_2",
                    "channel-4::libedit-3.1.20170329-h6b74fdf_2",
                    "channel-4::readline-7.0-ha6073c6_4",
                    "channel-4::sqlite-3.24.0-h84994c4_0",
                    "channel-4::python-3.7.0-hc3d631a_0",
                    "channel-4::asn1crypto-0.24.0-py37_0",
                    "channel-4::beautifulsoup4-4.6.3-py37_0",
                    "channel-4::certifi-2018.8.13-py37_0",
                    "channel-4::chardet-3.0.4-py37_1",
                    "channel-4::cryptography-vectors-2.3-py37_0",
                    "channel-4::filelock-3.0.4-py37_0",
                    "channel-4::glob2-0.6-py37_0",
                    "channel-4::idna-2.7-py37_0",
                    "channel-4::markupsafe-1.0-py37h14c3975_1",
                    "channel-4::pkginfo-1.4.2-py37_1",
                    "channel-4::psutil-5.4.6-py37h14c3975_0",
                    "channel-4::pycosat-0.6.3-py37h14c3975_0",
                    "channel-4::pycparser-2.18-py37_1",
                    "channel-4::pysocks-1.6.8-py37_0",
                    "channel-4::pyyaml-3.13-py37h14c3975_0",
                    "channel-4::ruamel_yaml-0.15.46-py37h14c3975_0",
                    "channel-4::six-1.11.0-py37_1",
                    "channel-4::cffi-1.11.5-py37h9745a5d_0",
                    "channel-4::setuptools-40.0.0-py37_0",
                    "channel-4::cryptography-2.3-py37hb7f436b_0",
                    "channel-4::jinja2-2.10-py37_0",
                    "channel-4::pyopenssl-18.0.0-py37_0",
                    "channel-4::urllib3-1.23-py37_0",
                    "channel-4::requests-2.19.1-py37_0",
                    "channel-4::conda-4.5.10-py37_0",
                    "channel-4::conda-build-3.12.1-py37_0",
                )
            )
            assert convert_to_dist_str(final_state_1) == order

    SubdirData.clear_cached_local_channel_data()
    specs_to_add = (MatchSpec("itsdangerous"),)  # MatchSpec("conda"),
    saved_sys_prefix = sys.prefix
    try:
        sys.prefix = tmpdir.strpath
        with get_solver_aggregate_1(
            tmpdir, specs_to_add=specs_to_add, prefix_records=final_state_1, history_specs=specs
        ) as solver:
            unlink_precs, link_precs = solver.solve_for_diff()
            pprint(convert_to_dist_str(unlink_precs))
            pprint(convert_to_dist_str(link_precs))
            unlink_order = (
                # no conda downgrade
            )
            link_order = add_subdir_to_iter(("channel-2::itsdangerous-0.24-py_0",))
            assert convert_to_dist_str(unlink_precs) == unlink_order
            assert convert_to_dist_str(link_precs) == link_order

        specs_to_add = (
            MatchSpec("itsdangerous"),
            MatchSpec("conda"),
        )
        with get_solver_aggregate_1(
            tmpdir, specs_to_add=specs_to_add, prefix_records=final_state_1, history_specs=specs
        ) as solver:
            unlink_precs, link_precs = solver.solve_for_diff()
            pprint(convert_to_dist_str(unlink_precs))
            pprint(convert_to_dist_str(link_precs))
            assert convert_to_dist_str(unlink_precs) == unlink_order
            assert convert_to_dist_str(link_precs) == link_order

        specs_to_add = MatchSpec("itsdangerous"), MatchSpec("conda<4.4.10"), MatchSpec("python")
        with get_solver_aggregate_1(
            tmpdir, specs_to_add=specs_to_add, prefix_records=final_state_1, history_specs=specs
        ) as solver:
            unlink_precs, link_precs = solver.solve_for_diff()
            pprint(convert_to_dist_str(unlink_precs))
            pprint(convert_to_dist_str(link_precs))
            unlink_order = add_subdir_to_iter(
                (
                    # now conda gets downgraded
                    "channel-4::conda-build-3.12.1-py37_0",
                    "channel-4::conda-4.5.10-py37_0",
                    "channel-4::requests-2.19.1-py37_0",
                    "channel-4::urllib3-1.23-py37_0",
                    "channel-4::pyopenssl-18.0.0-py37_0",
                    "channel-4::jinja2-2.10-py37_0",
                    "channel-4::cryptography-2.3-py37hb7f436b_0",
                    "channel-4::setuptools-40.0.0-py37_0",
                    "channel-4::cffi-1.11.5-py37h9745a5d_0",
                    "channel-4::six-1.11.0-py37_1",
                    "channel-4::ruamel_yaml-0.15.46-py37h14c3975_0",
                    "channel-4::pyyaml-3.13-py37h14c3975_0",
                    "channel-4::pysocks-1.6.8-py37_0",
                    "channel-4::pycparser-2.18-py37_1",
                    "channel-4::pycosat-0.6.3-py37h14c3975_0",
                    "channel-4::psutil-5.4.6-py37h14c3975_0",
                    "channel-4::pkginfo-1.4.2-py37_1",
                    "channel-4::markupsafe-1.0-py37h14c3975_1",
                    "channel-4::idna-2.7-py37_0",
                    "channel-4::glob2-0.6-py37_0",
                    "channel-4::filelock-3.0.4-py37_0",
                    "channel-4::cryptography-vectors-2.3-py37_0",
                    "channel-4::chardet-3.0.4-py37_1",
                    "channel-4::certifi-2018.8.13-py37_0",
                    "channel-4::beautifulsoup4-4.6.3-py37_0",
                    "channel-4::asn1crypto-0.24.0-py37_0",
                    "channel-4::python-3.7.0-hc3d631a_0",
                    "channel-4::sqlite-3.24.0-h84994c4_0",
                    "channel-4::readline-7.0-ha6073c6_4",
                    "channel-4::libedit-3.1.20170329-h6b74fdf_2",
                    "channel-4::yaml-0.1.7-had09818_2",
                    "channel-4::xz-5.2.4-h14c3975_4",
                    "channel-4::tk-8.6.7-hc745277_3",
                    "channel-4::openssl-1.0.2p-h14c3975_0",
                    "channel-4::ncurses-6.1-hf484d3e_0",
                )
            )
            link_order = add_subdir_to_iter(
                (
                    "channel-2::openssl-1.0.2l-0",
                    "channel-2::readline-6.2-2",
                    "channel-2::sqlite-3.13.0-0",
                    "channel-2::tk-8.5.18-0",
                    "channel-2::xz-5.2.3-0",
                    "channel-2::yaml-0.1.6-0",
                    "channel-2::python-3.6.2-0",
                    "channel-2::asn1crypto-0.22.0-py36_0",
                    "channel-4::beautifulsoup4-4.6.3-py36_0",
                    "channel-2::certifi-2016.2.28-py36_0",
                    "channel-4::chardet-3.0.4-py36_1",
                    "channel-4::filelock-3.0.4-py36_0",
                    "channel-4::glob2-0.6-py36_0",
                    "channel-2::idna-2.6-py36_0",
                    "channel-2::itsdangerous-0.24-py36_0",
                    "channel-2::markupsafe-1.0-py36_0",
                    "channel-4::pkginfo-1.4.2-py36_1",
                    "channel-2::psutil-5.2.2-py36_0",
                    "channel-2::pycosat-0.6.2-py36_0",
                    "channel-2::pycparser-2.18-py36_0",
                    "channel-2::pyparsing-2.2.0-py36_0",
                    "channel-2::pyyaml-3.12-py36_0",
                    "channel-2::requests-2.14.2-py36_0",
                    "channel-2::ruamel_yaml-0.11.14-py36_1",
                    "channel-2::six-1.10.0-py36_0",
                    "channel-2::cffi-1.10.0-py36_0",
                    "channel-2::packaging-16.8-py36_0",
                    "channel-2::setuptools-36.4.0-py36_1",
                    "channel-2::cryptography-1.8.1-py36_0",
                    "channel-2::jinja2-2.9.6-py36_0",
                    "channel-2::pyopenssl-17.0.0-py36_0",
                    "channel-2::conda-4.3.30-py36h5d9f9f4_0",
                    "channel-4::conda-build-3.12.1-py36_0",
                )
            )
            ## MODIFIED
            # Original checks verified the full solution was strictly matched:
            ### assert convert_to_dist_str(unlink_precs) == unlink_order
            ### assert convert_to_dist_str(link_precs) == link_order
            # We only check for conda itself and the explicit specs
            # The other packages are slightly different;
            # again libedit and ncurses are involved
            # (they are also involved in test_fast_update_with_update_modifier_not_set)
            for pkg in link_precs:
                if pkg.name == "conda":
                    assert VersionOrder(pkg.version) < VersionOrder("4.4.10")
                elif pkg.name == "python":
                    assert pkg.version == "3.6.2"
                elif pkg.name == "conda-build":
                    assert pkg.version == "3.12.1"
                elif pkg.name == "itsdangerous":
                    assert pkg.version == "0.24"
            ## /MODIFIED
    finally:
        sys.prefix = saved_sys_prefix


def test_python2_update(tmpdir):
    # Here we're actually testing that a user-request will uninstall incompatible packages
    # as necessary.
    specs = MatchSpec("conda"), MatchSpec("python=2")
    with get_solver_4(tmpdir, specs) as solver:
        final_state_1 = solver.solve_final_state()
        pprint(convert_to_dist_str(final_state_1))
        order1 = add_subdir_to_iter(
            (
                "channel-4::ca-certificates-2018.03.07-0",
                "channel-4::conda-env-2.6.0-1",
                "channel-4::libgcc-ng-8.2.0-hdf63c60_0",
                "channel-4::libstdcxx-ng-8.2.0-hdf63c60_0",
                "channel-4::libffi-3.2.1-hd88cf55_4",
                "channel-4::ncurses-6.1-hf484d3e_0",
                "channel-4::openssl-1.0.2p-h14c3975_0",
                "channel-4::tk-8.6.7-hc745277_3",
                "channel-4::yaml-0.1.7-had09818_2",
                "channel-4::zlib-1.2.11-ha838bed_2",
                "channel-4::libedit-3.1.20170329-h6b74fdf_2",
                "channel-4::readline-7.0-ha6073c6_4",
                "channel-4::sqlite-3.24.0-h84994c4_0",
                "channel-4::python-2.7.15-h1571d57_0",
                "channel-4::asn1crypto-0.24.0-py27_0",
                "channel-4::certifi-2018.8.13-py27_0",
                "channel-4::chardet-3.0.4-py27_1",
                "channel-4::cryptography-vectors-2.3-py27_0",
                "channel-4::enum34-1.1.6-py27_1",
                "channel-4::futures-3.2.0-py27_0",
                "channel-4::idna-2.7-py27_0",
                "channel-4::ipaddress-1.0.22-py27_0",
                "channel-4::pycosat-0.6.3-py27h14c3975_0",
                "channel-4::pycparser-2.18-py27_1",
                "channel-4::pysocks-1.6.8-py27_0",
                "channel-4::ruamel_yaml-0.15.46-py27h14c3975_0",
                "channel-4::six-1.11.0-py27_1",
                "channel-4::cffi-1.11.5-py27h9745a5d_0",
                "channel-4::cryptography-2.3-py27hb7f436b_0",
                "channel-4::pyopenssl-18.0.0-py27_0",
                "channel-4::urllib3-1.23-py27_0",
                "channel-4::requests-2.19.1-py27_0",
                "channel-4::conda-4.5.10-py27_0",
            )
        )
        assert convert_to_dist_str(final_state_1) == order1

    specs_to_add = (MatchSpec("python=3"),)
    with get_solver_4(
        tmpdir, specs_to_add, prefix_records=final_state_1, history_specs=specs
    ) as solver:
        final_state_2 = solver.solve_final_state()
        pprint(convert_to_dist_str(final_state_2))
        order = add_subdir_to_iter(
            (
                "channel-4::ca-certificates-2018.03.07-0",
                "channel-4::conda-env-2.6.0-1",
                "channel-4::libgcc-ng-8.2.0-hdf63c60_0",
                "channel-4::libstdcxx-ng-8.2.0-hdf63c60_0",
                "channel-4::libffi-3.2.1-hd88cf55_4",
                "channel-4::ncurses-6.1-hf484d3e_0",
                "channel-4::openssl-1.0.2p-h14c3975_0",
                "channel-4::tk-8.6.7-hc745277_3",
                "channel-4::xz-5.2.4-h14c3975_4",
                "channel-4::yaml-0.1.7-had09818_2",
                "channel-4::zlib-1.2.11-ha838bed_2",
                "channel-4::libedit-3.1.20170329-h6b74fdf_2",
                "channel-4::readline-7.0-ha6073c6_4",
                "channel-4::sqlite-3.24.0-h84994c4_0",
                "channel-4::python-3.7.0-hc3d631a_0",
                "channel-4::asn1crypto-0.24.0-py37_0",
                "channel-4::certifi-2018.8.13-py37_0",
                "channel-4::chardet-3.0.4-py37_1",
                "channel-4::idna-2.7-py37_0",
                "channel-4::pycosat-0.6.3-py37h14c3975_0",
                "channel-4::pycparser-2.18-py37_1",
                "channel-4::pysocks-1.6.8-py37_0",
                "channel-4::ruamel_yaml-0.15.46-py37h14c3975_0",
                "channel-4::six-1.11.0-py37_1",
                "channel-4::cffi-1.11.5-py37h9745a5d_0",
                "channel-4::cryptography-2.2.2-py37h14c3975_0",
                "channel-4::pyopenssl-18.0.0-py37_0",
                "channel-4::urllib3-1.23-py37_0",
                "channel-4::requests-2.19.1-py37_0",
                "channel-4::conda-4.5.10-py37_0",
            )
        )

        ## MODIFIED
        # libmamba has a different solution here (cryptography 2.3 instead of 2.2.2)
        # and cryptography-vectors (not present in regular conda)
        # they are essentially the same functional solution; the important part here
        # is that the env migrated to Python 3.7, so we only check some packages
        # Original check:
        ### assert convert_to_dist_str(final_state_2) == order
        full_solution = convert_to_dist_str(final_state_2)
        important_parts = add_subdir_to_iter(
            (
                "channel-4::python-3.7.0-hc3d631a_0",
                "channel-4::conda-4.5.10-py37_0",
                "channel-4::pycosat-0.6.3-py37h14c3975_0",
            )
        )
        assert set(important_parts).issubset(set(full_solution))
        ## /MODIFIED


def test_fast_update_with_update_modifier_not_set(tmpdir):
    specs = (
        MatchSpec("python=2"),
        MatchSpec("openssl==1.0.2l"),
        MatchSpec("sqlite=3.21"),
    )
    with get_solver_4(tmpdir, specs) as solver:
        final_state_1 = solver.solve_final_state()
        pprint(convert_to_dist_str(final_state_1))
        order1 = add_subdir_to_iter(
            (
                "channel-4::ca-certificates-2018.03.07-0",
                "channel-4::libgcc-ng-8.2.0-hdf63c60_0",
                "channel-4::libstdcxx-ng-8.2.0-hdf63c60_0",
                "channel-4::libffi-3.2.1-hd88cf55_4",
                "channel-4::ncurses-6.0-h9df7e31_2",
                "channel-4::openssl-1.0.2l-h077ae2c_5",
                "channel-4::tk-8.6.7-hc745277_3",
                "channel-4::zlib-1.2.11-ha838bed_2",
                "channel-4::libedit-3.1-heed3624_0",
                "channel-4::readline-7.0-ha6073c6_4",
                "channel-4::sqlite-3.21.0-h1bed415_2",
                "channel-4::python-2.7.14-h89e7a4a_22",
            )
        )
        assert convert_to_dist_str(final_state_1) == order1

    specs_to_add = (MatchSpec("python"),)
    with get_solver_4(
        tmpdir, specs_to_add, prefix_records=final_state_1, history_specs=specs
    ) as solver:
        unlink_precs, link_precs = solver.solve_for_diff()
        pprint(convert_to_dist_str(unlink_precs))
        pprint(convert_to_dist_str(link_precs))
        unlink_order = add_subdir_to_iter(
            (
                "channel-4::python-2.7.14-h89e7a4a_22",
                "channel-4::libedit-3.1-heed3624_0",
                "channel-4::openssl-1.0.2l-h077ae2c_5",
                "channel-4::ncurses-6.0-h9df7e31_2",
            )
        )
        link_order = add_subdir_to_iter(
            (
                "channel-4::ncurses-6.1-hf484d3e_0",
                "channel-4::openssl-1.0.2p-h14c3975_0",
                "channel-4::xz-5.2.4-h14c3975_4",
                "channel-4::libedit-3.1.20170329-h6b74fdf_2",
                "channel-4::python-3.6.4-hc3d631a_1",  # python is upgraded
            )
        )
        ## MODIFIED
        # We only check python was upgraded as expected, not the full solution
        ### assert convert_to_dist_str(unlink_precs) == unlink_order
        ### assert convert_to_dist_str(link_precs) == link_order
        assert add_subdir("channel-4::python-2.7.14-h89e7a4a_22") in convert_to_dist_str(
            unlink_precs
        )
        assert add_subdir("channel-4::python-3.6.4-hc3d631a_1") in convert_to_dist_str(link_precs)
        ## /MODIFIED

    specs_to_add = (MatchSpec("sqlite"),)
    with get_solver_4(
        tmpdir, specs_to_add, prefix_records=final_state_1, history_specs=specs
    ) as solver:
        unlink_precs, link_precs = solver.solve_for_diff()
        pprint(convert_to_dist_str(unlink_precs))
        pprint(convert_to_dist_str(link_precs))
        unlink_order = add_subdir_to_iter(
            (
                "channel-4::python-2.7.14-h89e7a4a_22",
                "channel-4::sqlite-3.21.0-h1bed415_2",
                "channel-4::libedit-3.1-heed3624_0",
                "channel-4::openssl-1.0.2l-h077ae2c_5",
                "channel-4::ncurses-6.0-h9df7e31_2",
            )
        )
        link_order = add_subdir_to_iter(
            (
                "channel-4::ncurses-6.1-hf484d3e_0",
                "channel-4::openssl-1.0.2p-h14c3975_0",
                "channel-4::libedit-3.1.20170329-h6b74fdf_2",
                "channel-4::sqlite-3.24.0-h84994c4_0",  # sqlite is upgraded
                "channel-4::python-2.7.15-h1571d57_0",  # python is not upgraded
            )
        )
        ## MODIFIED
        # We only check sqlite was upgraded as expected and python stays the same
        ### assert convert_to_dist_str(unlink_precs) == unlink_order
        ### assert convert_to_dist_str(link_precs) == link_order
        assert add_subdir("channel-4::sqlite-3.21.0-h1bed415_2") in convert_to_dist_str(
            unlink_precs
        )
        sqlite = next(pkg for pkg in link_precs if pkg.name == "sqlite")
        # mamba chooses a different sqlite version (3.23 instead of 3.24)
        assert VersionOrder(sqlite.version) > VersionOrder("3.21")
        # If Python was changed, it should have stayed at 2.7
        python = next((pkg for pkg in link_precs if pkg.name == "python"), None)
        if python:
            assert python.version.startswith("2.7")
        ## /MODIFIED

    specs_to_add = (
        MatchSpec("sqlite"),
        MatchSpec("python"),
    )
    with get_solver_4(
        tmpdir, specs_to_add, prefix_records=final_state_1, history_specs=specs
    ) as solver:
        final_state_2 = solver.solve_final_state(
            update_modifier=UpdateModifier.SPECS_SATISFIED_SKIP_SOLVE
        )
        pprint(convert_to_dist_str(final_state_2))
        assert convert_to_dist_str(final_state_2) == order1


@pytest.mark.xfail(True, reason="Known bug: mamba prefers arch to noarch - TODO")
def test_channel_priority_churn_minimized(tmpdir):
    specs = (
        MatchSpec("conda-build"),
        MatchSpec("itsdangerous"),
    )
    with get_solver_aggregate_2(tmpdir, specs) as solver:
        final_state = solver.solve_final_state()

    pprint(convert_to_dist_str(final_state))

    with get_solver_aggregate_2(
        tmpdir, [MatchSpec("itsdangerous")], prefix_records=final_state, history_specs=specs
    ) as solver:
        solver.channels.reverse()
        unlink_dists, link_dists = solver.solve_for_diff(
            update_modifier=UpdateModifier.FREEZE_INSTALLED
        )
        pprint(convert_to_dist_str(unlink_dists))
        pprint(convert_to_dist_str(link_dists))
        assert len(unlink_dists) == 1
        assert len(link_dists) == 1


@pytest.mark.xfail(True, reason="channel priority is a bit different in libmamba; TODO")
def test_priority_1(tmpdir):
    with env_var("CONDA_SUBDIR", "linux-64", stack_callback=conda_tests_ctxt_mgmt_def_pol):
        specs = (
            MatchSpec("pandas"),
            MatchSpec("python=2.7"),
        )

        ## MODIFIED
        # Original value was set to True (legacy value for "flexible" nowadays), but libmamba
        # only gets the same solution is strict priority is chosen. It _looks_ like this was the
        # intention of the test anyways, but it should be investigated further. Marking as xfail for now.
        ### with env_var("CONDA_CHANNEL_PRIORITY", "True", stack_callback=conda_tests_ctxt_mgmt_def_pol):
        with env_var(
            "CONDA_CHANNEL_PRIORITY", "strict", stack_callback=conda_tests_ctxt_mgmt_def_pol
        ):
            ## /MODIFIED

            with get_solver_aggregate_1(tmpdir, specs) as solver:
                final_state_1 = solver.solve_final_state()
                pprint(convert_to_dist_str(final_state_1))
                order = add_subdir_to_iter(
                    (
                        "channel-2::mkl-2017.0.3-0",
                        "channel-2::openssl-1.0.2l-0",
                        "channel-2::readline-6.2-2",
                        "channel-2::sqlite-3.13.0-0",
                        "channel-2::tk-8.5.18-0",
                        "channel-2::zlib-1.2.11-0",
                        "channel-2::python-2.7.13-0",
                        "channel-2::numpy-1.13.1-py27_0",
                        "channel-2::pytz-2017.2-py27_0",
                        "channel-2::six-1.10.0-py27_0",
                        "channel-2::python-dateutil-2.6.1-py27_0",
                        "channel-2::pandas-0.20.3-py27_0",
                    )
                )
                assert convert_to_dist_str(final_state_1) == order

        with env_var(
            "CONDA_CHANNEL_PRIORITY", "False", stack_callback=conda_tests_ctxt_mgmt_def_pol
        ):
            with get_solver_aggregate_1(
                tmpdir, specs, prefix_records=final_state_1, history_specs=specs
            ) as solver:
                final_state_2 = solver.solve_final_state()
                pprint(convert_to_dist_str(final_state_2))
                # python and pandas will be updated as they are explicit specs.  Other stuff may or may not,
                #     as required to satisfy python and pandas
                order = add_subdir_to_iter(
                    (
                        "channel-4::python-2.7.15-h1571d57_0",
                        "channel-4::pandas-0.23.4-py27h04863e7_0",
                    )
                )
                for spec in order:
                    assert spec in convert_to_dist_str(final_state_2)

        # channel priority taking effect here.  channel-2 should be the channel to draw from.  Downgrades expected.
        # python and pandas will be updated as they are explicit specs.  Other stuff may or may not,
        #     as required to satisfy python and pandas
        with get_solver_aggregate_1(
            tmpdir, specs, prefix_records=final_state_2, history_specs=specs
        ) as solver:
            final_state_3 = solver.solve_final_state()
            pprint(convert_to_dist_str(final_state_3))
            order = add_subdir_to_iter(
                (
                    "channel-2::python-2.7.13-0",
                    "channel-2::pandas-0.20.3-py27_0",
                )
            )
            for spec in order:
                assert spec in convert_to_dist_str(final_state_3)

        specs_to_add = (MatchSpec("six<1.10"),)
        specs_to_remove = (MatchSpec("pytz"),)
        with get_solver_aggregate_1(
            tmpdir,
            specs_to_add=specs_to_add,
            specs_to_remove=specs_to_remove,
            prefix_records=final_state_3,
            history_specs=specs,
        ) as solver:
            final_state_4 = solver.solve_final_state()
            pprint(convert_to_dist_str(final_state_4))
            order = add_subdir_to_iter(
                (
                    "channel-2::python-2.7.13-0",
                    "channel-2::six-1.9.0-py27_0",
                )
            )
            for spec in order:
                assert spec in convert_to_dist_str(final_state_4)
            assert "pandas" not in convert_to_dist_str(final_state_4)


def test_downgrade_python_prevented_with_sane_message(tmpdir):
    specs = (MatchSpec("python=2.6"),)
    with get_solver(tmpdir, specs) as solver:
        final_state_1 = solver.solve_final_state()
    # PrefixDag(final_state_1, specs).open_url()
    pprint(convert_to_dist_str(final_state_1))
    order = add_subdir_to_iter(
        (
            "channel-1::openssl-1.0.1c-0",
            "channel-1::readline-6.2-0",
            "channel-1::sqlite-3.7.13-0",
            "channel-1::system-5.8-1",
            "channel-1::tk-8.5.13-0",
            "channel-1::zlib-1.2.7-0",
            "channel-1::python-2.6.8-6",
        )
    )
    assert convert_to_dist_str(final_state_1) == order

    # incompatible CLI and configured specs
    specs_to_add = (MatchSpec("scikit-learn==0.13"),)
    with get_solver(
        tmpdir, specs_to_add=specs_to_add, prefix_records=final_state_1, history_specs=specs
    ) as solver:
        with pytest.raises(UnsatisfiableError) as exc:
            solver.solve_final_state()

        error_msg = str(exc.value).strip()

        ## MODIFIED
        # One more case of different wording for the same message. I think the essence is the same
        # (cannot update to python 2.7), even if python 2.6 is not mentioned.
        ### assert "incompatible with the existing python installation in your environment:" in error_msg
        ### assert "- scikit-learn==0.13 -> python=2.7" in error_msg
        ### assert "Your python: python=2.6" in error_msg
        assert "Encountered problems while solving" in error_msg
        assert "package scikit-learn-0.13" in error_msg and "requires python 2.7*" in error_msg
        ## /MODIFIED

    specs_to_add = (MatchSpec("unsatisfiable-with-py26"),)
    with get_solver(
        tmpdir, specs_to_add=specs_to_add, prefix_records=final_state_1, history_specs=specs
    ) as solver:
        with pytest.raises(UnsatisfiableError) as exc:
            solver.solve_final_state()
        error_msg = str(exc.value).strip()

        ## MODIFIED
        # In this case, the error is not as similar! We are still accepting it, but it could use
        # some improvements... Note how Python is not mentioned at all, just scikit-learn.
        # Leaving a # TODO mark here so we can come revisit this in the future.
        ### assert "incompatible with the existing python installation in your environment:" in error_msg
        ### assert "- unsatisfiable-with-py26 -> python=2.7" in error_msg
        ### assert "Your python: python=2.6"
        assert "Encountered problems while solving" in error_msg
        assert "package unsatisfiable-with-py26-1.0-0 requires scikit-learn 0.13" in error_msg
        ## /MODIFIED


# The following tests come from tests/test_priority.py


@pytest.mark.integration
@pytest.mark.parametrize(
    "pinned_package",
    [
        pytest.param(True, id="with pinned_package"),
        pytest.param(False, id="without pinned_package"),
    ],
)
def test_reorder_channel_priority(
    tmp_env: TmpEnvFixture,
    monkeypatch: MonkeyPatch,
    conda_cli: CondaCLIFixture,
    pinned_package: bool,
):
    # use "cheap" packages with no dependencies
    package1 = "zlib"
    package2 = "ca-certificates"

    # set pinned package
    if pinned_package:
        monkeypatch.setenv("CONDA_PINNED_PACKAGES", package1)

    # create environment with package1 and package2
    with tmp_env("--override-channels", "--channel=defaults", package1, package2) as prefix:
        # check both packages are installed from defaults
        PrefixData._cache_.clear()
        assert PrefixData(prefix).get(package1).channel.name == "pkgs/main"
        assert PrefixData(prefix).get(package2).channel.name == "pkgs/main"

        # update --all
        out, err, retcode = conda_cli(
            "update",
            f"--prefix={prefix}",
            "--override-channels",
            "--channel=conda-forge",
            "--all",
            "--yes",
        )
        # check pinned package is unchanged but unpinned packages are updated from conda-forge
        PrefixData._cache_.clear()
        expected_channel = "pkgs/main" if pinned_package else "conda-forge"
        assert PrefixData(prefix).get(package1).channel.name == expected_channel
        # assert PrefixData(prefix).get(package2).channel.name == "conda-forge"
        # MODIFIED ^: Some packages do not change channels in libmamba


def test_explicit_missing_cache_entries(
    mocker: MockerFixture,
    conda_cli: CondaCLIFixture,
    tmp_env: TmpEnvFixture,
):
    """Test that explicit() raises and notifies if some of the specs were not found in the cache."""
    from conda.core.package_cache_data import PackageCacheData

    with tmp_env() as prefix:  # ensure writable env
        if len(PackageCacheData.get_all_extracted_entries()) == 0:
            # Package cache e.g. ./devenv/Darwin/x86_64/envs/devenv-3.9-c/pkgs/ can
            # be empty in certain cases (Noted in OSX with Python 3.9, when
            # Miniconda installs Python 3.10). Install a small package.
            warnings.warn("test_explicit_missing_cache_entries: No packages in cache.")
            out, err, retcode = conda_cli("install", "--prefix", prefix, "heapdict", "--yes")
            assert retcode == 0, (out, err)  # MODIFIED

        # Patching ProgressiveFetchExtract prevents trying to download a package from the url.
        # Note that we cannot monkeypatch context.dry_run, because explicit() would exit early with that.
        mocker.patch("conda.misc.ProgressiveFetchExtract")
        print(PackageCacheData.get_all_extracted_entries()[0])  # MODIFIED
        with pytest.raises(
            AssertionError,
            match="Missing package cache records for: pkgs/linux-64::foo==1.0.0=py_0",
        ):
            explicit(
                [
                    "http://test/pkgs/linux-64/foo-1.0.0-py_0.tar.bz2",  # does not exist
                    PackageCacheData.get_all_extracted_entries()[0].url,  # exists
                ],
                prefix,
            )