import sys
import json
from subprocess import check_output
from conda.testing.integration import _get_temp_prefix


def test_channel_matchspec():
    out = check_output(
        [
            sys.executable,
            "-m",
            "conda",
            "create",
            "-p",
            _get_temp_prefix(),
            "--experimental-solver=libmamba",
            "--json",
            "--override-channels",
            "-c",
            "defaults",
            "conda-forge::libblas=*=*openblas",
            "python=3.9",
        ]
    )
    result = json.loads(out)
    assert result["success"] is True
    for record in result["actions"]["LINK"]:
        if record["name"] == "numpy":
            assert record["channel"] == "conda-forge"
        elif record["name"] == "python":
            assert record["channel"] == "pkgs/main"
