"""
This module fixes some tests found across conda/conda's suite to
check the "spirit" of the test, instead of making explicit comparisons
in stdout messages, overly strict solver checks and other differences
that do not result in incompatible behaviour.

We are copying those offending tests instead of patching them to keep
conda/conda code base as unaffected by this work as possible, but it is
indeed feasible to upgrade those tests in the future for more flexible
comparisons. This is only a workaround during the experimental phase.

Tests were brought over and patched on Feb 7th, 2022, following the
source found in commit 98fb262c610e17a7731b9183bf37cca98dcc1a71.
"""

import os
import re
import sys
from pprint import pprint

import pytest

from conda.auxlib.ish import dals
from conda.common.io import env_var
from conda.base.constants import UpdateModifier
from conda.base.context import context ,conda_tests_ctxt_mgmt_def_pol
from conda.core.package_cache_data import PackageCacheData
from conda.core.prefix_data import PrefixData
from conda.exceptions import UnsatisfiableError, SpecsConfigurationConflictError
from conda.models.match_spec import MatchSpec
from conda.testing.cases import BaseTestCase
from conda.testing.integration import (
    run_command,
    Commands,
    make_temp_env,
    package_is_installed,
)
from conda.testing.helpers import (
    add_subdir_to_iter,
    convert_to_dist_str,
    get_solver,
    get_solver_2,
    get_solver_4,
    get_solver_cuda,
    get_solver_aggregate_1,
    get_solver_aggregate_2,
)


@pytest.mark.integration
class PatchedCondaTestCreate(BaseTestCase):
    """
    These tests come from `conda/conda::tests/test_create.py`
    """

    def setUp(self):
        PackageCacheData.clear()

    # https://github.com/conda/conda/issues/9124
    @pytest.mark.skipif(context.subdir != 'linux-64', reason="lazy; package constraint here only valid on linux-64")
    def test_neutering_of_historic_specs(self):
        with make_temp_env('psutil=5.6.3=py37h7b6447c_0') as prefix:
            stdout, stderr, _ = run_command(Commands.INSTALL, prefix, "python=3.6")
            with open(os.path.join(prefix, 'conda-meta', 'history')) as f:
                d = f.read()

            ### MODIFIED
            ## libmamba relaxes more aggressively sometimes
            ## instead of relaxing from pkgname=version=build to pkgname=version, it
            ## goes to just pkgname; this is because libmamba does not take into account
            ## matchspec target and optionality (iow, MatchSpec.conda_build_form() does not)
            ## Original check was stricter:
            # assert re.search(r"neutered specs:.*'psutil==5.6.3'\]", d)
            assert re.search(r"neutered specs:.*'psutil'\]", d)
            ### /MODIFIED

            # this would be unsatisfiable if the neutered specs were not being factored in correctly.
            #    If this command runs successfully (does not raise), then all is well.
            stdout, stderr, _ = run_command(Commands.INSTALL, prefix, "imagesize")

    def test_pinned_override_with_explicit_spec(self):
        with make_temp_env("python=3.6") as prefix:

            ### MODIFIED
            ## Original test assumed the `python=3.6` spec above resolves to `python=3.6.5`
            ## Instead we only pin whatever the solver decided to install
            ## Original lines were:
            # run_command(Commands.CONFIG, prefix,
            #             "--add", "pinned_packages", "python=3.6.5")
            python = next(PrefixData(prefix).query("python"))
            run_command(Commands.CONFIG, prefix,
                        "--add", "pinned_packages", f"python={python.version}")
            ### /MODIFIED

            run_command(Commands.INSTALL, prefix, "python=3.7", no_capture=True)
            assert package_is_installed(prefix, "python=3.7")


# The following tests come from `conda/conda::tests/core/test_solve.py`

@pytest.mark.integration
def test_pinned_1(tmpdir):
    specs = MatchSpec("numpy"),
    with get_solver(tmpdir, specs) as solver:
        final_state_1 = solver.solve_final_state()
        # PrefixDag(final_state_1, specs).open_url()
        pprint(convert_to_dist_str(final_state_1))
        order = add_subdir_to_iter((
            'channel-1::openssl-1.0.1c-0',
            'channel-1::readline-6.2-0',
            'channel-1::sqlite-3.7.13-0',
            'channel-1::system-5.8-1',
            'channel-1::tk-8.5.13-0',
            'channel-1::zlib-1.2.7-0',
            'channel-1::python-3.3.2-0',
            'channel-1::numpy-1.7.1-py33_0',
        ))
        assert convert_to_dist_str(final_state_1) == order

    with env_var("CONDA_PINNED_PACKAGES", "python=2.6&iopro<=1.4.2", stack_callback=conda_tests_ctxt_mgmt_def_pol):
        specs = MatchSpec("system=5.8=0"),
        with get_solver(tmpdir, specs) as solver:
            final_state_1 = solver.solve_final_state()
            # PrefixDag(final_state_1, specs).open_url()
            pprint(convert_to_dist_str(final_state_1))
            order = add_subdir_to_iter((
                'channel-1::system-5.8-0',
            ))
            assert convert_to_dist_str(final_state_1) == order

        # ignore_pinned=True
        specs_to_add = MatchSpec("python"),
        with get_solver(tmpdir, specs_to_add=specs_to_add, prefix_records=final_state_1,
                        history_specs=specs) as solver:
            final_state_2 = solver.solve_final_state(ignore_pinned=True)
            # PrefixDag(final_state_1, specs).open_url()
            pprint(convert_to_dist_str(final_state_2))
            order = add_subdir_to_iter((
                'channel-1::openssl-1.0.1c-0',
                'channel-1::readline-6.2-0',
                'channel-1::sqlite-3.7.13-0',
                'channel-1::system-5.8-0',
                'channel-1::tk-8.5.13-0',
                'channel-1::zlib-1.2.7-0',
                'channel-1::python-3.3.2-0',
            ))
            assert convert_to_dist_str(final_state_2) == order

        # ignore_pinned=False
        specs_to_add = MatchSpec("python"),
        with get_solver(tmpdir, specs_to_add=specs_to_add, prefix_records=final_state_1,
                        history_specs=specs) as solver:
            final_state_2 = solver.solve_final_state(ignore_pinned=False)
            # PrefixDag(final_state_1, specs).open_url()
            pprint(convert_to_dist_str(final_state_2))
            order = add_subdir_to_iter((
                'channel-1::openssl-1.0.1c-0',
                'channel-1::readline-6.2-0',
                'channel-1::sqlite-3.7.13-0',
                'channel-1::system-5.8-0',
                'channel-1::tk-8.5.13-0',
                'channel-1::zlib-1.2.7-0',
                'channel-1::python-2.6.8-6',
            ))
            assert convert_to_dist_str(final_state_2) == order

        # incompatible CLI and configured specs
        specs_to_add = MatchSpec("scikit-learn==0.13"),
        with get_solver(tmpdir, specs_to_add=specs_to_add, prefix_records=final_state_1,
                        history_specs=specs) as solver:
            with pytest.raises(SpecsConfigurationConflictError) as exc:
                solver.solve_final_state(ignore_pinned=False)
            kwargs = exc.value._kwargs
            assert kwargs["requested_specs"] == ["scikit-learn==0.13"]
            assert kwargs["pinned_specs"] == ["python=2.6"]

        specs_to_add = MatchSpec("numba"),
        history_specs = MatchSpec("python"), MatchSpec("system=5.8=0"),
        with get_solver(tmpdir, specs_to_add=specs_to_add, prefix_records=final_state_2,
                        history_specs=history_specs) as solver:
            final_state_3 = solver.solve_final_state()
            # PrefixDag(final_state_1, specs).open_url()
            pprint(convert_to_dist_str(final_state_3))
            order = add_subdir_to_iter((
                'channel-1::openssl-1.0.1c-0',
                'channel-1::readline-6.2-0',
                'channel-1::sqlite-3.7.13-0',
                'channel-1::system-5.8-0',
                'channel-1::tk-8.5.13-0',
                'channel-1::zlib-1.2.7-0',
                'channel-1::llvm-3.2-0',
                'channel-1::python-2.6.8-6',
                'channel-1::argparse-1.2.1-py26_0',
                'channel-1::llvmpy-0.11.2-py26_0',
                'channel-1::numpy-1.7.1-py26_0',
                'channel-1::numba-0.8.1-np17py26_0',
            ))
            assert convert_to_dist_str(final_state_3) == order

        specs_to_add = MatchSpec("python"),
        history_specs = MatchSpec("python"), MatchSpec("system=5.8=0"), MatchSpec("numba"),
        with get_solver(tmpdir, specs_to_add=specs_to_add, prefix_records=final_state_3,
                        history_specs=history_specs) as solver:
            final_state_4 = solver.solve_final_state(update_modifier=UpdateModifier.UPDATE_DEPS)
            # PrefixDag(final_state_1, specs).open_url()
            pprint(convert_to_dist_str(final_state_4))
            order = add_subdir_to_iter((
                'channel-1::openssl-1.0.1c-0',
                'channel-1::readline-6.2-0',
                'channel-1::sqlite-3.7.13-0',
                'channel-1::system-5.8-1',
                'channel-1::tk-8.5.13-0',
                'channel-1::zlib-1.2.7-0',
                'channel-1::llvm-3.2-0',
                'channel-1::python-2.6.8-6',
                'channel-1::argparse-1.2.1-py26_0',
                'channel-1::llvmpy-0.11.2-py26_0',
                'channel-1::numpy-1.7.1-py26_0',
                'channel-1::numba-0.8.1-np17py26_0',
            ))
            assert convert_to_dist_str(final_state_4) == order

        specs_to_add = MatchSpec("python"),
        history_specs = MatchSpec("python"), MatchSpec("system=5.8=0"), MatchSpec("numba"),
        with get_solver(tmpdir, specs_to_add=specs_to_add, prefix_records=final_state_4,
                        history_specs=history_specs) as solver:
            final_state_5 = solver.solve_final_state(update_modifier=UpdateModifier.UPDATE_ALL)
            # PrefixDag(final_state_1, specs).open_url()
            pprint(convert_to_dist_str(final_state_5))
            order = add_subdir_to_iter((
                'channel-1::openssl-1.0.1c-0',
                'channel-1::readline-6.2-0',
                'channel-1::sqlite-3.7.13-0',
                'channel-1::system-5.8-1',
                'channel-1::tk-8.5.13-0',
                'channel-1::zlib-1.2.7-0',
                'channel-1::llvm-3.2-0',
                'channel-1::python-2.6.8-6',
                'channel-1::argparse-1.2.1-py26_0',
                'channel-1::llvmpy-0.11.2-py26_0',
                'channel-1::numpy-1.7.1-py26_0',
                'channel-1::numba-0.8.1-np17py26_0',
            ))
            assert convert_to_dist_str(final_state_5) == order

    # now update without pinning
    specs_to_add = MatchSpec("python"),
    history_specs = MatchSpec("python"), MatchSpec("system=5.8=0"), MatchSpec("numba"),
    with get_solver(tmpdir, specs_to_add=specs_to_add, prefix_records=final_state_4,
                    history_specs=history_specs) as solver:
        final_state_5 = solver.solve_final_state(update_modifier=UpdateModifier.UPDATE_ALL)
        # PrefixDag(final_state_1, specs).open_url()
        print(convert_to_dist_str(final_state_5))
        order = add_subdir_to_iter((
            'channel-1::openssl-1.0.1c-0',
            'channel-1::readline-6.2-0',
            'channel-1::sqlite-3.7.13-0',
            'channel-1::system-5.8-1',
            'channel-1::tk-8.5.13-0',
            'channel-1::zlib-1.2.7-0',
            'channel-1::llvm-3.2-0',
            'channel-1::python-3.3.2-0',
            'channel-1::llvmpy-0.11.2-py33_0',
            'channel-1::numpy-1.7.1-py33_0',
            'channel-1::numba-0.8.1-np17py33_0',
        ))
        assert convert_to_dist_str(final_state_5) == order


@pytest.mark.integration
def test_freeze_deps_1(tmpdir):
    specs = MatchSpec("six=1.7"),
    with get_solver_2(tmpdir, specs) as solver:
        final_state_1 = solver.solve_final_state()
        pprint(convert_to_dist_str(final_state_1))
        order = add_subdir_to_iter((
            'channel-2::openssl-1.0.2l-0',
            'channel-2::readline-6.2-2',
            'channel-2::sqlite-3.13.0-0',
            'channel-2::tk-8.5.18-0',
            'channel-2::xz-5.2.3-0',
            'channel-2::zlib-1.2.11-0',
            'channel-2::python-3.4.5-0',
            'channel-2::six-1.7.3-py34_0',
        ))
        assert convert_to_dist_str(final_state_1) == order

    specs_to_add = MatchSpec("bokeh"),
    with get_solver_2(tmpdir, specs_to_add, prefix_records=final_state_1, history_specs=specs) as solver:
        unlink_precs, link_precs = solver.solve_for_diff()
        pprint(convert_to_dist_str(unlink_precs))
        pprint(convert_to_dist_str(link_precs))
        unlink_order = ()
        link_order = add_subdir_to_iter((
            'channel-2::mkl-2017.0.3-0',
            'channel-2::yaml-0.1.6-0',
            'channel-2::backports_abc-0.5-py34_0',
            'channel-2::markupsafe-1.0-py34_0',
            'channel-2::numpy-1.13.0-py34_0',
            'channel-2::pyyaml-3.12-py34_0',
            'channel-2::requests-2.14.2-py34_0',
            'channel-2::setuptools-27.2.0-py34_0',
            'channel-2::jinja2-2.9.6-py34_0',
            'channel-2::python-dateutil-2.6.1-py34_0',
            'channel-2::tornado-4.4.2-py34_0',
            'channel-2::bokeh-0.12.4-py34_0',
        ))
        assert convert_to_dist_str(unlink_precs) == unlink_order
        assert convert_to_dist_str(link_precs) == link_order

    # now we can't install the latest bokeh 0.12.5, but instead we get bokeh 0.12.4
    specs_to_add = MatchSpec("bokeh"),
    with get_solver_2(tmpdir, specs_to_add, prefix_records=final_state_1,
                    history_specs=(MatchSpec("six=1.7"), MatchSpec("python=3.4"))) as solver:
        unlink_precs, link_precs = solver.solve_for_diff()
        pprint(convert_to_dist_str(unlink_precs))
        pprint(convert_to_dist_str(link_precs))
        unlink_order = ()
        link_order = add_subdir_to_iter((
            'channel-2::mkl-2017.0.3-0',
            'channel-2::yaml-0.1.6-0',
            'channel-2::backports_abc-0.5-py34_0',
            'channel-2::markupsafe-1.0-py34_0',
            'channel-2::numpy-1.13.0-py34_0',
            'channel-2::pyyaml-3.12-py34_0',
            'channel-2::requests-2.14.2-py34_0',
            'channel-2::setuptools-27.2.0-py34_0',
            'channel-2::jinja2-2.9.6-py34_0',
            'channel-2::python-dateutil-2.6.1-py34_0',
            'channel-2::tornado-4.4.2-py34_0',
            'channel-2::bokeh-0.12.4-py34_0',
        ))
        assert convert_to_dist_str(unlink_precs) == unlink_order
        assert convert_to_dist_str(link_precs) == link_order

    # here, the python=3.4 spec can't be satisfied, so it's dropped, and we go back to py27
    with pytest.raises(UnsatisfiableError):
        specs_to_add = MatchSpec("bokeh=0.12.5"),
        with get_solver_2(tmpdir, specs_to_add, prefix_records=final_state_1,
                        history_specs=(MatchSpec("six=1.7"), MatchSpec("python=3.4"))) as solver:
            unlink_precs, link_precs = solver.solve_for_diff()

    # adding the explicit python spec allows conda to change the python versions.
    # one possible outcome is that this updates to python 3.6.  That is not desirable because of the
    #    explicit "six=1.7" request in the history.  It should only neuter that spec if there's no way
    #    to solve it with that spec.
    specs_to_add = MatchSpec("bokeh=0.12.5"), MatchSpec("python")
    with get_solver_2(tmpdir, specs_to_add, prefix_records=final_state_1,
                    history_specs=(MatchSpec("six=1.7"), MatchSpec("python=3.4"))) as solver:
        unlink_precs, link_precs = solver.solve_for_diff()
        pprint(convert_to_dist_str(unlink_precs))
        pprint(convert_to_dist_str(link_precs))
        unlink_order = add_subdir_to_iter((
            'channel-2::six-1.7.3-py34_0',
            'channel-2::python-3.4.5-0',
            'channel-2::xz-5.2.3-0',
        ))
        link_order = add_subdir_to_iter((
            'channel-2::mkl-2017.0.3-0',
            'channel-2::yaml-0.1.6-0',
            'channel-2::python-2.7.13-0',
            'channel-2::backports-1.0-py27_0',
            'channel-2::backports_abc-0.5-py27_0',
            'channel-2::certifi-2016.2.28-py27_0',
            'channel-2::futures-3.1.1-py27_0',
            'channel-2::markupsafe-1.0-py27_0',
            'channel-2::numpy-1.13.1-py27_0',
            'channel-2::pyyaml-3.12-py27_0',
            'channel-2::requests-2.14.2-py27_0',
            'channel-2::six-1.7.3-py27_0',
            'channel-2::python-dateutil-2.6.1-py27_0',
            'channel-2::setuptools-36.4.0-py27_1',
            'channel-2::singledispatch-3.4.0.3-py27_0',
            'channel-2::ssl_match_hostname-3.5.0.1-py27_0',
            'channel-2::jinja2-2.9.6-py27_0',
            'channel-2::tornado-4.5.2-py27_0',
            'channel-2::bokeh-0.12.5-py27_1',
        ))
        assert convert_to_dist_str(unlink_precs) == unlink_order
        assert convert_to_dist_str(link_precs) == link_order

    # here, the python=3.4 spec can't be satisfied, so it's dropped, and we go back to py27
    specs_to_add = MatchSpec("bokeh=0.12.5"),
    with get_solver_2(tmpdir, specs_to_add, prefix_records=final_state_1,
                    history_specs=(MatchSpec("six=1.7"), MatchSpec("python=3.4"))) as solver:
        with pytest.raises(UnsatisfiableError):
            solver.solve_final_state(update_modifier=UpdateModifier.FREEZE_INSTALLED)


def test_cuda_fail_1(tmpdir):
    specs = MatchSpec("cudatoolkit"),

    # No cudatoolkit in index for CUDA 8.0
    with env_var('CONDA_OVERRIDE_CUDA', '8.0'):
        with get_solver_cuda(tmpdir, specs) as solver:
            with pytest.raises(UnsatisfiableError) as exc:
                final_state = solver.solve_final_state()

    if sys.platform == "darwin":
        plat = "osx-64"
    elif sys.platform == "linux":
        plat = "linux-64"
    elif sys.platform == "win32":
        if platform.architecture()[0] == "32bit":
            plat = "win-32"
        else:
            plat = "win-64"
    else:
        plat = "linux-64"

    assert str(exc.value).strip() == dals("""The following specifications were found to be incompatible with your system:

  - feature:/{}::__cuda==8.0=0
  - cudatoolkit -> __cuda[version='>=10.0|>=9.0']

Your installed version is: 8.0""".format(plat))


def test_cuda_fail_2(tmpdir):
    specs = MatchSpec("cudatoolkit"),

    # No CUDA on system
    with env_var('CONDA_OVERRIDE_CUDA', ''):
        with get_solver_cuda(tmpdir, specs) as solver:
            with pytest.raises(UnsatisfiableError) as exc:
                final_state = solver.solve_final_state()

    assert str(exc.value).strip() == dals("""The following specifications were found to be incompatible with your system:

  - cudatoolkit -> __cuda[version='>=10.0|>=9.0']

Your installed version is: not available""")


def test_update_all_1(tmpdir):
    specs = MatchSpec("numpy=1.5"), MatchSpec("python=2.6"), MatchSpec("system[version=*,build_number=0]")
    with get_solver(tmpdir, specs) as solver:
        final_state_1 = solver.solve_final_state()
        # PrefixDag(final_state_1, specs).open_url()
        print(convert_to_dist_str(final_state_1))
        order = add_subdir_to_iter((
            'channel-1::openssl-1.0.1c-0',
            'channel-1::readline-6.2-0',
            'channel-1::sqlite-3.7.13-0',
            'channel-1::system-5.8-0',
            'channel-1::tk-8.5.13-0',
            'channel-1::zlib-1.2.7-0',
            'channel-1::python-2.6.8-6',
            'channel-1::numpy-1.5.1-py26_4',
        ))
        assert convert_to_dist_str(final_state_1) == order

    specs_to_add = MatchSpec("numba=0.6"), MatchSpec("numpy")
    with get_solver(tmpdir, specs_to_add, prefix_records=final_state_1, history_specs=specs) as solver:
        final_state_2 = solver.solve_final_state()
        # PrefixDag(final_state_2, specs).open_url()
        print(convert_to_dist_str(final_state_2))
        order = add_subdir_to_iter((
            'channel-1::openssl-1.0.1c-0',
            'channel-1::readline-6.2-0',
            'channel-1::sqlite-3.7.13-0',
            'channel-1::system-5.8-0',
            'channel-1::tk-8.5.13-0',
            'channel-1::zlib-1.2.7-0',
            'channel-1::llvm-3.2-0',
            'channel-1::python-2.6.8-6',
            'channel-1::llvmpy-0.10.2-py26_0',
            'channel-1::nose-1.3.0-py26_0',
            'channel-1::numpy-1.7.1-py26_0',
            'channel-1::numba-0.6.0-np17py26_0',
        ))
        assert convert_to_dist_str(final_state_2) == order

    specs_to_add = MatchSpec("numba=0.6"),
    with get_solver(tmpdir, specs_to_add, prefix_records=final_state_1, history_specs=specs) as solver:
        final_state_2 = solver.solve_final_state(update_modifier=UpdateModifier.UPDATE_ALL)
        # PrefixDag(final_state_2, specs).open_url()
        print(convert_to_dist_str(final_state_2))
        order = add_subdir_to_iter((
            'channel-1::openssl-1.0.1c-0',
            'channel-1::readline-6.2-0',
            'channel-1::sqlite-3.7.13-0',
            'channel-1::system-5.8-1',
            'channel-1::tk-8.5.13-0',
            'channel-1::zlib-1.2.7-0',
            'channel-1::llvm-3.2-0',
            'channel-1::python-2.6.8-6',  # stick with python=2.6 even though UPDATE_ALL
            'channel-1::llvmpy-0.10.2-py26_0',
            'channel-1::nose-1.3.0-py26_0',
            'channel-1::numpy-1.7.1-py26_0',
            'channel-1::numba-0.6.0-np17py26_0',
        ))
        assert convert_to_dist_str(final_state_2) == order

def test_conda_downgrade(tmpdir):
    specs = MatchSpec("conda-build"),
    with env_var("CONDA_CHANNEL_PRIORITY", "False", stack_callback=conda_tests_ctxt_mgmt_def_pol):
        with get_solver_aggregate_1(tmpdir, specs) as solver:
            final_state_1 = solver.solve_final_state()
            pprint(convert_to_dist_str(final_state_1))
            order = add_subdir_to_iter((
                'channel-4::ca-certificates-2018.03.07-0',
                'channel-2::conda-env-2.6.0-0',
                'channel-2::libffi-3.2.1-1',
                'channel-4::libgcc-ng-8.2.0-hdf63c60_0',
                'channel-4::libstdcxx-ng-8.2.0-hdf63c60_0',
                'channel-2::zlib-1.2.11-0',
                'channel-4::ncurses-6.1-hf484d3e_0',
                'channel-4::openssl-1.0.2p-h14c3975_0',
                'channel-4::patchelf-0.9-hf484d3e_2',
                'channel-4::tk-8.6.7-hc745277_3',
                'channel-4::xz-5.2.4-h14c3975_4',
                'channel-4::yaml-0.1.7-had09818_2',
                'channel-4::libedit-3.1.20170329-h6b74fdf_2',
                'channel-4::readline-7.0-ha6073c6_4',
                'channel-4::sqlite-3.24.0-h84994c4_0',
                'channel-4::python-3.7.0-hc3d631a_0',
                'channel-4::asn1crypto-0.24.0-py37_0',
                'channel-4::beautifulsoup4-4.6.3-py37_0',
                'channel-4::certifi-2018.8.13-py37_0',
                'channel-4::chardet-3.0.4-py37_1',
                'channel-4::cryptography-vectors-2.3-py37_0',
                'channel-4::filelock-3.0.4-py37_0',
                'channel-4::glob2-0.6-py37_0',
                'channel-4::idna-2.7-py37_0',
                'channel-4::markupsafe-1.0-py37h14c3975_1',
                'channel-4::pkginfo-1.4.2-py37_1',
                'channel-4::psutil-5.4.6-py37h14c3975_0',
                'channel-4::pycosat-0.6.3-py37h14c3975_0',
                'channel-4::pycparser-2.18-py37_1',
                'channel-4::pysocks-1.6.8-py37_0',
                'channel-4::pyyaml-3.13-py37h14c3975_0',
                'channel-4::ruamel_yaml-0.15.46-py37h14c3975_0',
                'channel-4::six-1.11.0-py37_1',
                'channel-4::cffi-1.11.5-py37h9745a5d_0',
                'channel-4::setuptools-40.0.0-py37_0',
                'channel-4::cryptography-2.3-py37hb7f436b_0',
                'channel-4::jinja2-2.10-py37_0',
                'channel-4::pyopenssl-18.0.0-py37_0',
                'channel-4::urllib3-1.23-py37_0',
                'channel-4::requests-2.19.1-py37_0',
                'channel-4::conda-4.5.10-py37_0',
                'channel-4::conda-build-3.12.1-py37_0'
            ))
            assert convert_to_dist_str(final_state_1) == order

    specs_to_add = MatchSpec("itsdangerous"),  # MatchSpec("conda"),
    saved_sys_prefix = sys.prefix
    try:
        sys.prefix = tmpdir.strpath
        with get_solver_aggregate_1(tmpdir, specs_to_add=specs_to_add, prefix_records=final_state_1,
                                    history_specs=specs) as solver:
            unlink_precs, link_precs = solver.solve_for_diff()
            pprint(convert_to_dist_str(unlink_precs))
            pprint(convert_to_dist_str(link_precs))
            unlink_order = (
                # no conda downgrade
            )
            link_order = add_subdir_to_iter((
                'channel-2::itsdangerous-0.24-py_0',
            ))
            assert convert_to_dist_str(unlink_precs) == unlink_order
            assert convert_to_dist_str(link_precs) == link_order

        specs_to_add = MatchSpec("itsdangerous"), MatchSpec("conda"),
        with get_solver_aggregate_1(tmpdir, specs_to_add=specs_to_add, prefix_records=final_state_1,
                                    history_specs=specs) as solver:
            unlink_precs, link_precs = solver.solve_for_diff()
            pprint(convert_to_dist_str(unlink_precs))
            pprint(convert_to_dist_str(link_precs))
            assert convert_to_dist_str(unlink_precs) == unlink_order
            assert convert_to_dist_str(link_precs) == link_order

        specs_to_add = MatchSpec("itsdangerous"), MatchSpec("conda<4.4.10"), MatchSpec("python")
        with get_solver_aggregate_1(tmpdir, specs_to_add=specs_to_add, prefix_records=final_state_1,
                                    history_specs=specs) as solver:
            unlink_precs, link_precs = solver.solve_for_diff()
            pprint(convert_to_dist_str(unlink_precs))
            pprint(convert_to_dist_str(link_precs))
            unlink_order = add_subdir_to_iter((
                # now conda gets downgraded
                'channel-4::conda-build-3.12.1-py37_0',
                'channel-4::conda-4.5.10-py37_0',
                'channel-4::requests-2.19.1-py37_0',
                'channel-4::urllib3-1.23-py37_0',
                'channel-4::pyopenssl-18.0.0-py37_0',
                'channel-4::jinja2-2.10-py37_0',
                'channel-4::cryptography-2.3-py37hb7f436b_0',
                'channel-4::setuptools-40.0.0-py37_0',
                'channel-4::cffi-1.11.5-py37h9745a5d_0',
                'channel-4::six-1.11.0-py37_1',
                'channel-4::ruamel_yaml-0.15.46-py37h14c3975_0',
                'channel-4::pyyaml-3.13-py37h14c3975_0',
                'channel-4::pysocks-1.6.8-py37_0',
                'channel-4::pycparser-2.18-py37_1',
                'channel-4::pycosat-0.6.3-py37h14c3975_0',
                'channel-4::psutil-5.4.6-py37h14c3975_0',
                'channel-4::pkginfo-1.4.2-py37_1',
                'channel-4::markupsafe-1.0-py37h14c3975_1',
                'channel-4::idna-2.7-py37_0',
                'channel-4::glob2-0.6-py37_0',
                'channel-4::filelock-3.0.4-py37_0',
                'channel-4::cryptography-vectors-2.3-py37_0',
                'channel-4::chardet-3.0.4-py37_1',
                'channel-4::certifi-2018.8.13-py37_0',
                'channel-4::beautifulsoup4-4.6.3-py37_0',
                'channel-4::asn1crypto-0.24.0-py37_0',
                'channel-4::python-3.7.0-hc3d631a_0',
                'channel-4::sqlite-3.24.0-h84994c4_0',
                'channel-4::readline-7.0-ha6073c6_4',
                'channel-4::libedit-3.1.20170329-h6b74fdf_2',
                'channel-4::yaml-0.1.7-had09818_2',
                'channel-4::xz-5.2.4-h14c3975_4',
                'channel-4::tk-8.6.7-hc745277_3',
                'channel-4::openssl-1.0.2p-h14c3975_0',
                'channel-4::ncurses-6.1-hf484d3e_0',
            ))
            link_order = add_subdir_to_iter((
                'channel-2::openssl-1.0.2l-0',
                'channel-2::readline-6.2-2',
                'channel-2::sqlite-3.13.0-0',
                'channel-2::tk-8.5.18-0',
                'channel-2::xz-5.2.3-0',
                'channel-2::yaml-0.1.6-0',
                'channel-2::python-3.6.2-0',
                'channel-2::asn1crypto-0.22.0-py36_0',
                'channel-4::beautifulsoup4-4.6.3-py36_0',
                'channel-2::certifi-2016.2.28-py36_0',
                'channel-4::chardet-3.0.4-py36_1',
                'channel-4::filelock-3.0.4-py36_0',
                'channel-4::glob2-0.6-py36_0',
                'channel-2::idna-2.6-py36_0',
                'channel-2::itsdangerous-0.24-py36_0',
                'channel-2::markupsafe-1.0-py36_0',
                'channel-4::pkginfo-1.4.2-py36_1',
                'channel-2::psutil-5.2.2-py36_0',
                'channel-2::pycosat-0.6.2-py36_0',
                'channel-2::pycparser-2.18-py36_0',
                'channel-2::pyparsing-2.2.0-py36_0',
                'channel-2::pyyaml-3.12-py36_0',
                'channel-2::requests-2.14.2-py36_0',
                'channel-2::ruamel_yaml-0.11.14-py36_1',
                'channel-2::six-1.10.0-py36_0',
                'channel-2::cffi-1.10.0-py36_0',
                'channel-2::packaging-16.8-py36_0',
                'channel-2::setuptools-36.4.0-py36_1',
                'channel-2::cryptography-1.8.1-py36_0',
                'channel-2::jinja2-2.9.6-py36_0',
                'channel-2::pyopenssl-17.0.0-py36_0',
                'channel-2::conda-4.3.30-py36h5d9f9f4_0',
                'channel-4::conda-build-3.12.1-py36_0'
            ))
            assert convert_to_dist_str(unlink_precs) == unlink_order
            assert convert_to_dist_str(link_precs) == link_order
    finally:
        sys.prefix = saved_sys_prefix

def test_python2_update(tmpdir):
    # Here we're actually testing that a user-request will uninstall incompatible packages
    # as necessary.
    specs = MatchSpec("conda"), MatchSpec("python=2")
    with get_solver_4(tmpdir, specs) as solver:
        final_state_1 = solver.solve_final_state()
        pprint(convert_to_dist_str(final_state_1))
        order1 = add_subdir_to_iter((
            'channel-4::ca-certificates-2018.03.07-0',
            'channel-4::conda-env-2.6.0-1',
            'channel-4::libgcc-ng-8.2.0-hdf63c60_0',
            'channel-4::libstdcxx-ng-8.2.0-hdf63c60_0',
            'channel-4::libffi-3.2.1-hd88cf55_4',
            'channel-4::ncurses-6.1-hf484d3e_0',
            'channel-4::openssl-1.0.2p-h14c3975_0',
            'channel-4::tk-8.6.7-hc745277_3',
            'channel-4::yaml-0.1.7-had09818_2',
            'channel-4::zlib-1.2.11-ha838bed_2',
            'channel-4::libedit-3.1.20170329-h6b74fdf_2',
            'channel-4::readline-7.0-ha6073c6_4',
            'channel-4::sqlite-3.24.0-h84994c4_0',
            'channel-4::python-2.7.15-h1571d57_0',
            'channel-4::asn1crypto-0.24.0-py27_0',
            'channel-4::certifi-2018.8.13-py27_0',
            'channel-4::chardet-3.0.4-py27_1',
            'channel-4::cryptography-vectors-2.3-py27_0',
            'channel-4::enum34-1.1.6-py27_1',
            'channel-4::futures-3.2.0-py27_0',
            'channel-4::idna-2.7-py27_0',
            'channel-4::ipaddress-1.0.22-py27_0',
            'channel-4::pycosat-0.6.3-py27h14c3975_0',
            'channel-4::pycparser-2.18-py27_1',
            'channel-4::pysocks-1.6.8-py27_0',
            'channel-4::ruamel_yaml-0.15.46-py27h14c3975_0',
            'channel-4::six-1.11.0-py27_1',
            'channel-4::cffi-1.11.5-py27h9745a5d_0',
            'channel-4::cryptography-2.3-py27hb7f436b_0',
            'channel-4::pyopenssl-18.0.0-py27_0',
            'channel-4::urllib3-1.23-py27_0',
            'channel-4::requests-2.19.1-py27_0',
            'channel-4::conda-4.5.10-py27_0',
        ))
        assert convert_to_dist_str(final_state_1) == order1

    specs_to_add = MatchSpec("python=3"),
    with get_solver_4(tmpdir, specs_to_add, prefix_records=final_state_1, history_specs=specs) as solver:
        final_state_2 = solver.solve_final_state()
        pprint(convert_to_dist_str(final_state_2))
        order = add_subdir_to_iter((
            'channel-4::ca-certificates-2018.03.07-0',
            'channel-4::conda-env-2.6.0-1',
            'channel-4::libgcc-ng-8.2.0-hdf63c60_0',
            'channel-4::libstdcxx-ng-8.2.0-hdf63c60_0',
            'channel-4::libffi-3.2.1-hd88cf55_4',
            'channel-4::ncurses-6.1-hf484d3e_0',
            'channel-4::openssl-1.0.2p-h14c3975_0',
            'channel-4::tk-8.6.7-hc745277_3',
            'channel-4::xz-5.2.4-h14c3975_4',
            'channel-4::yaml-0.1.7-had09818_2',
            'channel-4::zlib-1.2.11-ha838bed_2',
            'channel-4::libedit-3.1.20170329-h6b74fdf_2',
            'channel-4::readline-7.0-ha6073c6_4',
            'channel-4::sqlite-3.24.0-h84994c4_0',
            'channel-4::python-3.7.0-hc3d631a_0',
            'channel-4::asn1crypto-0.24.0-py37_0',
            'channel-4::certifi-2018.8.13-py37_0',
            'channel-4::chardet-3.0.4-py37_1',
            'channel-4::idna-2.7-py37_0',
            'channel-4::pycosat-0.6.3-py37h14c3975_0',
            'channel-4::pycparser-2.18-py37_1',
            'channel-4::pysocks-1.6.8-py37_0',
            'channel-4::ruamel_yaml-0.15.46-py37h14c3975_0',
            'channel-4::six-1.11.0-py37_1',
            'channel-4::cffi-1.11.5-py37h9745a5d_0',
            'channel-4::cryptography-2.2.2-py37h14c3975_0',
            'channel-4::pyopenssl-18.0.0-py37_0',
            'channel-4::urllib3-1.23-py37_0',
            'channel-4::requests-2.19.1-py37_0',
            'channel-4::conda-4.5.10-py37_0',
        ))
        assert convert_to_dist_str(final_state_2) == order


def test_fast_update_with_update_modifier_not_set(tmpdir):
    specs = MatchSpec("python=2"), MatchSpec("openssl==1.0.2l"), MatchSpec("sqlite=3.21"),
    with get_solver_4(tmpdir, specs) as solver:
        final_state_1 = solver.solve_final_state()
        pprint(convert_to_dist_str(final_state_1))
        order1 = add_subdir_to_iter((
            'channel-4::ca-certificates-2018.03.07-0',
            'channel-4::libgcc-ng-8.2.0-hdf63c60_0',
            'channel-4::libstdcxx-ng-8.2.0-hdf63c60_0',
            'channel-4::libffi-3.2.1-hd88cf55_4',
            'channel-4::ncurses-6.0-h9df7e31_2',
            'channel-4::openssl-1.0.2l-h077ae2c_5',
            'channel-4::tk-8.6.7-hc745277_3',
            'channel-4::zlib-1.2.11-ha838bed_2',
            'channel-4::libedit-3.1-heed3624_0',
            'channel-4::readline-7.0-ha6073c6_4',
            'channel-4::sqlite-3.21.0-h1bed415_2',
            'channel-4::python-2.7.14-h89e7a4a_22',
        ))
        assert convert_to_dist_str(final_state_1) == order1

    specs_to_add = MatchSpec("python"),
    with get_solver_4(tmpdir, specs_to_add, prefix_records=final_state_1, history_specs=specs) as solver:
        unlink_precs, link_precs = solver.solve_for_diff()
        pprint(convert_to_dist_str(unlink_precs))
        pprint(convert_to_dist_str(link_precs))
        unlink_order = add_subdir_to_iter((
            'channel-4::python-2.7.14-h89e7a4a_22',
            'channel-4::libedit-3.1-heed3624_0',
            'channel-4::openssl-1.0.2l-h077ae2c_5',
            'channel-4::ncurses-6.0-h9df7e31_2'
        ))
        link_order = add_subdir_to_iter((
            'channel-4::ncurses-6.1-hf484d3e_0',
            'channel-4::openssl-1.0.2p-h14c3975_0',
            'channel-4::xz-5.2.4-h14c3975_4',
            'channel-4::libedit-3.1.20170329-h6b74fdf_2',
            'channel-4::python-3.6.4-hc3d631a_1',  # python is upgraded
        ))
        assert convert_to_dist_str(unlink_precs) == unlink_order
        assert convert_to_dist_str(link_precs) == link_order

    specs_to_add = MatchSpec("sqlite"),
    with get_solver_4(tmpdir, specs_to_add, prefix_records=final_state_1, history_specs=specs) as solver:
        unlink_precs, link_precs = solver.solve_for_diff()
        pprint(convert_to_dist_str(unlink_precs))
        pprint(convert_to_dist_str(link_precs))
        unlink_order = add_subdir_to_iter((
            'channel-4::python-2.7.14-h89e7a4a_22',
            'channel-4::sqlite-3.21.0-h1bed415_2',
            'channel-4::libedit-3.1-heed3624_0',
            'channel-4::openssl-1.0.2l-h077ae2c_5',
            'channel-4::ncurses-6.0-h9df7e31_2',
        ))
        link_order = add_subdir_to_iter((
            'channel-4::ncurses-6.1-hf484d3e_0',
            'channel-4::openssl-1.0.2p-h14c3975_0',
            'channel-4::libedit-3.1.20170329-h6b74fdf_2',
            'channel-4::sqlite-3.24.0-h84994c4_0',  # sqlite is upgraded
            'channel-4::python-2.7.15-h1571d57_0',  # python is not upgraded
        ))
        assert convert_to_dist_str(unlink_precs) == unlink_order
        assert convert_to_dist_str(link_precs) == link_order

    specs_to_add = MatchSpec("sqlite"), MatchSpec("python"),
    with get_solver_4(tmpdir, specs_to_add, prefix_records=final_state_1, history_specs=specs) as solver:
        final_state_2 = solver.solve_final_state(update_modifier=UpdateModifier.SPECS_SATISFIED_SKIP_SOLVE)
        pprint(convert_to_dist_str(final_state_2))
        assert convert_to_dist_str(final_state_2) == order1


def test_channel_priority_churn_minimized(tmpdir):
    specs = MatchSpec("conda-build"), MatchSpec("itsdangerous"),
    with get_solver_aggregate_2(tmpdir, specs) as solver:
        final_state = solver.solve_final_state()

    pprint(convert_to_dist_str(final_state))

    with get_solver_aggregate_2(tmpdir, [MatchSpec('itsdangerous')],
                                prefix_records=final_state, history_specs=specs) as solver:
        solver.channels.reverse()
        unlink_dists, link_dists = solver.solve_for_diff(
            update_modifier=UpdateModifier.FREEZE_INSTALLED)
        pprint(convert_to_dist_str(unlink_dists))
        pprint(convert_to_dist_str(link_dists))
        assert len(unlink_dists) == 1
        assert len(link_dists) == 1

def test_priority_1(tmpdir):
    with env_var("CONDA_SUBDIR", "linux-64", stack_callback=conda_tests_ctxt_mgmt_def_pol):
        specs = MatchSpec("pandas"), MatchSpec("python=2.7"),
        with env_var("CONDA_CHANNEL_PRIORITY", "True", stack_callback=conda_tests_ctxt_mgmt_def_pol):
            with get_solver_aggregate_1(tmpdir, specs) as solver:
                final_state_1 = solver.solve_final_state()
                pprint(convert_to_dist_str(final_state_1))
                order = add_subdir_to_iter((
                    'channel-2::mkl-2017.0.3-0',
                    'channel-2::openssl-1.0.2l-0',
                    'channel-2::readline-6.2-2',
                    'channel-2::sqlite-3.13.0-0',
                    'channel-2::tk-8.5.18-0',
                    'channel-2::zlib-1.2.11-0',
                    'channel-2::python-2.7.13-0',
                    'channel-2::numpy-1.13.1-py27_0',
                    'channel-2::pytz-2017.2-py27_0',
                    'channel-2::six-1.10.0-py27_0',
                    'channel-2::python-dateutil-2.6.1-py27_0',
                    'channel-2::pandas-0.20.3-py27_0',
                ))
                assert convert_to_dist_str(final_state_1) == order

        with env_var("CONDA_CHANNEL_PRIORITY", "False", stack_callback=conda_tests_ctxt_mgmt_def_pol):
            with get_solver_aggregate_1(tmpdir, specs, prefix_records=final_state_1,
                                        history_specs=specs) as solver:
                final_state_2 = solver.solve_final_state()
                pprint(convert_to_dist_str(final_state_2))
                # python and pandas will be updated as they are explicit specs.  Other stuff may or may not,
                #     as required to satisfy python and pandas
                order = add_subdir_to_iter((
                    'channel-4::python-2.7.15-h1571d57_0',
                    'channel-4::pandas-0.23.4-py27h04863e7_0',
                ))
                for spec in order:
                    assert spec in convert_to_dist_str(final_state_2)

        # channel priority taking effect here.  channel-2 should be the channel to draw from.  Downgrades expected.
        # python and pandas will be updated as they are explicit specs.  Other stuff may or may not,
        #     as required to satisfy python and pandas
        with get_solver_aggregate_1(tmpdir, specs, prefix_records=final_state_2,
                                    history_specs=specs) as solver:
            final_state_3 = solver.solve_final_state()
            pprint(convert_to_dist_str(final_state_3))
            order = add_subdir_to_iter((
                'channel-2::python-2.7.13-0',
                'channel-2::pandas-0.20.3-py27_0',
            ))
            for spec in order:
                assert spec in convert_to_dist_str(final_state_3)

        specs_to_add = MatchSpec("six<1.10"),
        specs_to_remove = MatchSpec("pytz"),
        with get_solver_aggregate_1(tmpdir, specs_to_add=specs_to_add, specs_to_remove=specs_to_remove,
                                    prefix_records=final_state_3, history_specs=specs) as solver:
            final_state_4 = solver.solve_final_state()
            pprint(convert_to_dist_str(final_state_4))
            order = add_subdir_to_iter((
                'channel-2::python-2.7.13-0',
                'channel-2::six-1.9.0-py27_0',
            ))
            for spec in order:
                assert spec in convert_to_dist_str(final_state_4)
            assert 'pandas' not in convert_to_dist_str(final_state_4)


def test_downgrade_python_prevented_with_sane_message(tmpdir):
    specs = MatchSpec("python=2.6"),
    with get_solver(tmpdir, specs) as solver:
        final_state_1 = solver.solve_final_state()
    # PrefixDag(final_state_1, specs).open_url()
    pprint(convert_to_dist_str(final_state_1))
    order = add_subdir_to_iter((
        'channel-1::openssl-1.0.1c-0',
        'channel-1::readline-6.2-0',
        'channel-1::sqlite-3.7.13-0',
        'channel-1::system-5.8-1',
        'channel-1::tk-8.5.13-0',
        'channel-1::zlib-1.2.7-0',
        'channel-1::python-2.6.8-6',
    ))
    assert convert_to_dist_str(final_state_1) == order

    # incompatible CLI and configured specs
    specs_to_add = MatchSpec("scikit-learn==0.13"),
    with get_solver(tmpdir, specs_to_add=specs_to_add, prefix_records=final_state_1,
                    history_specs=specs) as solver:
        with pytest.raises(UnsatisfiableError) as exc:
            solver.solve_final_state()

        error_msg = str(exc.value).strip()
        assert "incompatible with the existing python installation in your environment:" in error_msg
        assert "- scikit-learn==0.13 -> python=2.7" in error_msg
        assert "Your python: python=2.6" in error_msg

    specs_to_add = MatchSpec("unsatisfiable-with-py26"),
    with get_solver(tmpdir, specs_to_add=specs_to_add, prefix_records=final_state_1,
                    history_specs=specs) as solver:
        with pytest.raises(UnsatisfiableError) as exc:
            solver.solve_final_state()
        error_msg = str(exc.value).strip()
        assert "incompatible with the existing python installation in your environment:" in error_msg
        assert "- unsatisfiable-with-py26 -> python=2.7" in error_msg
        assert "Your python: python=2.6"
