import sys
from subprocess import check_call


def test_matchspec_star_version():
    """
    Specs like `libblas=*=*mkl` choked on `MatchSpec.conda_build_form()`.
    We work around that with `.utils.safe_conda_build_form()`.
    Reported in https://github.com/conda/conda/issues/11347
    """
    check_call(
        [
            sys.executable,
            "-m",
            "conda",
            "create",
            "-p",
            "UNUSED",
            "--dry-run",
            "--override-channels",
            "-c",
            "conda-test",
            "--experimental-solver=libmamba",
            "activate_deactivate_package=*=*0",
        ]
    )
