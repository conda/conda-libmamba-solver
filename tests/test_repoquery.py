import json

from .utils import conda_subprocess

def test_repoquery():
    p = conda_subprocess("repoquery", "--help")
    assert "whoneeds" in p.stdout
    assert "depends" in p.stdout
    assert "search" in p.stdout

    p = conda_subprocess("repoquery", "depends", "conda", "--json")
    print(p.stdout)
    data = json.loads(p.stdout)
    assert data["result"]["status"] == "OK"
    assert len(data["result"]["pkgs"]) > 0
    assert len([p for p in data["result"]["pkgs"] if p["name"] == "python"]) == 1
