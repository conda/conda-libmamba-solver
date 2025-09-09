# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
import json

from conda.models.channel import Channel

from conda_libmamba_solver.index import LibMambaIndexHelper

from .utils import conda_subprocess, python_site_packages_path_support


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


def test_query_search():
    index = LibMambaIndexHelper(channels=[Channel("conda-forge")])
    for query in (
        "ca-certificates",
        "ca-certificates =2022.9.24",
        "ca-certificates >=2022.9.24",
        "ca-certificates >2022.9.24",
        "ca-certificates<2022.9.24,>2020",
        "ca-certificates<=2022.9.24,>2020",
        "ca-certificates !=2022.9.24,>2020",
        "ca-certificates=*=*_0",
        # TODO: channel specs are accepted but they seem to be ignored by libmambapy.Query!
        # "defaults::ca-certificates",
        # "defaults::ca-certificates=2022.9.24",
        # "defaults::ca-certificates[version='>=2022.9.24']",
        # "defaults::ca-certificates[build='*_0']",
    ):
        results = index.search(query)
        assert len(results) > 0, query

    assert index.search("ca-certificates=*=*_0") == index.search("ca-certificates[build='*_0']")
    assert index.search("ca-certificates >=2022.9.24") == index.search(
        "ca-certificates[version='>=2022.9.24']"
    )


def test_query_search_includes_python_site_packages_path():
    index = LibMambaIndexHelper(channels=[Channel("conda-forge")], subdirs=("linux-64", "noarch"))
    results = index.search("python=3.13.2=h4724d56_1_cp313t")
    assert len(results) == 1
    prec = results[0]
    assert prec.name == "python"
    assert prec.version == "3.13.2"
    if python_site_packages_path_support:
        assert prec.python_site_packages_path == "lib/python3.13t/site-packages"
    else:
        assert prec.python_site_packages_path is None
