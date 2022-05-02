import sys
from subprocess import check_call, check_output
import json


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


def test_build_string_filters():
    output = check_output(
        [
            sys.executable,
            "-m",
            "conda",
            "create",
            "-p",
            "UNUSED",
            "--dry-run",
            "--experimental-solver=libmamba",
            "numpy=*=*py38*",
            "--json",
        ]
    )
    data = json.loads(output)
    assert data["success"]
    for pkg in data["actions"]["LINK"]:
        if pkg["name"] == "python":
            assert pkg["version"].startswith("3.8")
        if pkg["name"] == "numpy":
            assert "py38" in pkg["build_string"]
