# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
Test sharded repodata.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.parse
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import msgpack
import pytest
import requests.exceptions
import zstandard
from conda.base.context import context, reset_context
from conda.core.subdir_data import SubdirData
from conda.gateways.connection.session import get_session
from conda.models.channel import Channel

from conda_libmamba_solver import shards, shards_cache, shards_subset
from conda_libmamba_solver.index import (
    LibMambaIndexHelper,
    _is_sharded_repodata_enabled,
    _package_info_from_package_dict,
)
from conda_libmamba_solver.shards import (
    ShardLike,
    Shards,
    _shards_connections,
    batch_retrieve_from_cache,
    fetch_shards_index,
    shard_mentioned_packages_2,
)
from conda_libmamba_solver.shards_subset import (
    Node,
    build_repodata_subset,
    fetch_channels,
)
from tests.channel_testing.helpers import _dummy_http_server

if TYPE_CHECKING:
    from conda_libmamba_solver.shards_typing import ShardDict, ShardsIndexDict

HERE = Path(__file__).parent


def package_names(shard: shards_cache.ShardDict):
    """
    All package names mentioned in a shard (should be a single package name)
    """
    return set(package["name"] for package in shard["packages"].values()) | set(
        package["name"] for package in shard["packages.conda"].values()
    )


def repodata_subset_size(channel_data):
    """
    Measure the size of a repodata subset as serialized to JSON. Discard data.
    """
    repodata_size = 0
    for _, shardlike in channel_data.items():
        repodata = shardlike.build_repodata()
        repodata_text = json.dumps(
            repodata, indent=0, separators=(",", ":"), sort_keys=True, ensure_ascii=False
        )
        repodata_size += len(repodata_text.encode("utf-8"))

    return repodata_size


@contextmanager
def _timer(name: str):
    begin = time.monotonic_ns()
    yield
    end = time.monotonic_ns()
    print(f"{name} took {(end - begin) / 1e9:0.6f}s")


@pytest.fixture
def prepare_shards_test(monkeypatch: pytest.MonkeyPatch):
    """
    Reset token to avoid being logged in. e.g. the testing channel doesn't understand them.
    Enable shards.
    """
    logging.basicConfig(level=logging.INFO)
    for module in (shards, shards_cache, shards_subset):
        module.log.setLevel(logging.DEBUG)

    monkeypatch.setenv("CONDA_TOKEN", "")
    monkeypatch.setenv("CONDA_PLUGINS_USE_SHARDED_REPODATA", "1")
    reset_context()
    assert _is_sharded_repodata_enabled()


FAKE_SHARD: ShardDict = {
    "packages": {
        "foo": {
            "name": "foo",
            "version": "1",
            "build": "0_a",
            "build_number": 0,
            "depends": ["bar", "baz"],
        }
    },
    "packages.conda": {
        "foo": {
            "name": "foo",
            "version": "1",
            "build": "0_a",
            "build_number": 0,
            "depends": ["quux", "warble"],
            "constrains": ["splat<3"],
            "sha256": hashlib.sha256().digest(),
        }
    },
}


@pytest.fixture
def http_server_shards(xprocess, tmp_path_factory):
    """
    A shard repository with a difference.
    """
    shards_repository = tmp_path_factory.mktemp("sharded_repo")
    noarch = shards_repository / "noarch"
    noarch.mkdir()

    foo_shard = zstandard.compress(msgpack.dumps(FAKE_SHARD))  # type: ignore
    foo_shard_digest = hashlib.sha256(foo_shard).digest()
    (noarch / f"{foo_shard_digest.hex()}.msgpack.zst").write_bytes(foo_shard)

    malformed = {"follows_schema": False}
    bad_schema = zstandard.compress(msgpack.dumps(malformed))  # type: ignore
    # XXX not-zstandard; not msgpack
    malformed_digest = hashlib.sha256(bad_schema).digest()

    (noarch / f"{malformed_digest.hex()}.msgpack.zst").write_bytes(bad_schema)
    not_zstd = b"not zstd"
    (noarch / f"{hashlib.sha256(not_zstd).digest().hex()}.msgpack.zst").write_bytes(not_zstd)
    not_msgpack = zstandard.compress(b"not msgpack")
    (noarch / f"{hashlib.sha256(not_msgpack).digest().hex()}.msgpack.zst").write_bytes(not_msgpack)
    fake_shards: ShardsIndexDict = {
        "info": {"subdir": "noarch", "base_url": "", "shards_base_url": ""},
        "version": 1,
        "shards": {
            "foo": foo_shard_digest,
            "wrong_package_name": foo_shard_digest,
            "fake_package": b"",
            "malformed": hashlib.sha256(bad_schema).digest(),
            "not_zstd": hashlib.sha256(not_zstd).digest(),
            "not_msgpack": hashlib.sha256(not_msgpack).digest(),
        },
    }
    (shards_repository / "noarch" / "repodata_shards.msgpack.zst").write_bytes(
        zstandard.compress(msgpack.dumps(fake_shards))  # type: ignore
    )
    yield from _dummy_http_server(
        xprocess, name="http_server_auth_none", port=0, auth="none", path=shards_repository
    )


def test_fetch_shards_error(http_server_shards):
    channel = Channel.from_url(f"{http_server_shards}/noarch")
    subdir_data = SubdirData(channel)
    found = fetch_shards_index(subdir_data)
    assert found

    not_found = fetch_shards_index(SubdirData(Channel.from_url(f"{http_server_shards}/linux-64")))
    assert not not_found

    # cover "unexpected package name in shard" branch
    found.visited.clear()
    assert "packages" in found.fetch_shard("wrong_package_name")

    # one non-error shard
    shard_a = found.fetch_shard("foo")
    shard_b = found.fetch_shard("foo")
    assert shard_a is shard_b
    found.visited.clear()  # force sqlite3 cache usage
    shard_c = found.fetch_shard("foo")
    assert shard_a is not shard_c
    assert shard_a == shard_c

    with pytest.raises(requests.exceptions.HTTPError):
        # XXX this is currently trying to decompress the server's 404 response, which should be a `requests.Response.raise_for_status()`
        found.fetch_shard("fake_package")

    # currently logs KeyError: 'packages', doesn't cache, returns decoded msgpack
    malo = found.fetch_shard("malformed")
    assert malo == {"follows_schema": False}  # XXX should we return None or raise

    with pytest.raises(zstandard.ZstdError):
        found.fetch_shard("not_zstd")

    # unclear if all possible "bad msgpack" errors inherit from a common class
    # besides ValueError
    with pytest.raises(ValueError):
        found.fetch_shard("not_msgpack")


def test_shards_base_url():
    shards = Shards(
        {
            "info": {
                "subdir": "noarch",
                "base_url": "",
                "shards_base_url": "https://shards.example.com/channel-name",
            },
            "version": 1,
            "shards": {"fake_package": b""},
        },
        "https://conda.anaconda.org/channel-name/noarch/",
        None,  # type: ignore
    )

    assert (
        shards.shard_url("fake_package") == "https://shards.example.com/channel-name/.msgpack.zst"
    )

    shards.shards_index["info"]["shards_base_url"] = ""

    assert (
        shards.shard_url("fake_package")
        == "https://conda.anaconda.org/channel-name/noarch/.msgpack.zst"
    )

    # no-trailing-/ example from prefix.dev metadata
    shards.url = "https://prefix.dev/conda-forge/osx-arm64/repodata_shards.msgpack.zst"
    shards.shards_index["info"]["base_url"] = "https://prefix.dev/conda-forge/osx-arm64"
    # shards_base_url should be suitable for string concatenation
    assert shards.shards_base_url == "https://prefix.dev/conda-forge/osx-arm64/"
    assert (
        shards.shard_url("fake_package") == "https://prefix.dev/conda-forge/osx-arm64/.msgpack.zst"
    )

    # relative shards_base_url
    shards.shards_index["info"]["shards_base_url"] = "./shards/"
    assert shards.shards_base_url == "https://prefix.dev/conda-forge/osx-arm64/shards/"

    # relative shards_base_url, with parent directory (not likely in the wild)
    shards.shards_index["info"]["shards_base_url"] = "../shards"
    assert shards.shards_base_url == "https://prefix.dev/conda-forge/shards/"


def test_shard_mentioned_packages_2():
    assert set(shard_mentioned_packages_2(FAKE_SHARD)) == set(
        (
            "bar",
            "baz",
            "quux",
            # "splat", # omit constrains
            "warble",
        )
    )

    # check that the bytes hash was converted to hex
    assert FAKE_SHARD["packages.conda"]["foo"]["sha256"] == hashlib.sha256().hexdigest()  # type: ignore


def test_fetch_shards_channels(prepare_shards_test: None):
    """
    Test all channels fetch as Shards or ShardLike, depending on availability.
    """
    channels = list(context.default_channels)
    print(channels)

    channels.append(Channel("conda-forge-sharded"))

    channel_data = fetch_channels(channels)

    # at least one should be real shards, not repodata.json presented as shards.
    assert any(isinstance(channel, Shards) for channel in channel_data.values())


def test_shard_cache(tmp_path: Path):
    cache = shards_cache.ShardCache(tmp_path)

    fake_shard = {"foo": "bar"}
    annotated_shard = shards_cache.AnnotatedRawShard(
        "https://foo",
        "foo",
        zstandard.compress(msgpack.dumps(fake_shard)),  # type: ignore
    )
    cache.insert(annotated_shard)

    data = cache.retrieve(annotated_shard.url)
    assert data == fake_shard
    assert data is not fake_shard

    data2 = cache.retrieve("notfound")
    assert data2 is None

    assert (tmp_path / shards_cache.SHARD_CACHE_NAME).exists()


def test_shard_cache_multiple(tmp_path: Path):
    """
    Test that retrieve_multiple() is equivalent to several retrieve() calls.
    """
    NUM_FAKE_SHARDS = 64

    cache = shards_cache.ShardCache(tmp_path)
    fake_shards = []

    compressor = zstandard.ZstdCompressor(level=1)
    for i in range(NUM_FAKE_SHARDS):
        fake_shard = {f"foo{i}": "bar"}
        annotated_shard = shards_cache.AnnotatedRawShard(
            f"https://foo{i}",
            f"foo{i}",
            compressor.compress(msgpack.dumps(fake_shard)),  # type: ignore
        )
        cache.insert(annotated_shard)
        fake_shards.append(annotated_shard)

    start_multiple = time.monotonic_ns()
    retrieved = cache.retrieve_multiple([shard.url for shard in fake_shards])
    end_multiple = time.monotonic_ns()

    assert len(retrieved) == NUM_FAKE_SHARDS

    print(
        f"retrieve {len(fake_shards)} shards in a single call: {(end_multiple - start_multiple) / 1e9:0.6f}s"
    )

    start_single = time.monotonic_ns()
    for i, url in enumerate([shard.url for shard in fake_shards]):
        single = cache.retrieve(url)
        assert retrieved[url] == single
    end_single = time.monotonic_ns()
    print(
        f"retrieve {len(fake_shards)} shards with multiple calls: {(end_single - start_single) / 1e9:0.6f}s"
    )

    if (end_single - start_single) != 0:  # avoid ZeroDivisionError
        print(
            f"Multiple API takes {(end_multiple - start_multiple) / (end_single - start_single):.2f} times as long."
        )

        assert (end_multiple - start_multiple) / (end_single - start_single) < 1, (
            "batch API took longer"
        )

    assert (tmp_path / shards_cache.SHARD_CACHE_NAME).exists()


def test_shard_cache_2():
    """
    Check that existing shards cache has been populated with matching shards,
    package names.
    """
    subdir_data = SubdirData(Channel("main/noarch"))
    # accepts directory that cache goes into, not database filename
    cache = shards_cache.ShardCache(Path(subdir_data.cache_path_base).parent)
    extant = {}
    for row in cache.conn.execute("SELECT url, package, shard, timestamp FROM shards"):
        extant[row["url"]] = shards_cache.AnnotatedRawShard(
            url=row["url"], package=row["package"], compressed_shard=row["shard"]
        )

    # check that url's are in the url field, and package names are in the package name field
    for url in extant:
        if shard := cache.retrieve(url):
            assert "://" in extant[url].url
            assert "://" not in extant[url].package
            for group in ("packages", "packages.conda"):
                assert all(
                    extant[url].package in (package["name"], "wrong_package_name")
                    for package in shard[group].values()
                ), f"Bad package name {extant[url].package}"


def test_shardlike():
    """
    ShardLike class presents repodata.json as shards in a way that is suitable
    for our subsetting algorithm.
    """
    repodata = json.loads(
        (Path(__file__).parent / "data" / "mamba_repo" / "noarch" / "repodata.json").read_text()
    )

    bad_repodata = repodata.copy()
    bad_repodata["info"] = {**bad_repodata["info"], "base_url": 4}
    with pytest.raises(TypeError):
        ShardLike(bad_repodata)

    # make fake packages
    for n in range(10):
        for m in range(n):  # 0 test0's
            repodata["packages"][f"test{n}{m}.tar.bz2"] = {"name": f"test{n}"}
            repodata["packages.conda"][f"test{n}{m}.tar.bz2"] = {"name": f"test{n}"}

    as_shards = ShardLike(repodata)

    assert len(as_shards.repodata_no_packages)
    assert len(as_shards.shards)

    assert sorted(as_shards.shards["test4"]["packages"].keys()) == [
        "test40.tar.bz2",
        "test41.tar.bz2",
        "test42.tar.bz2",
        "test43.tar.bz2",
    ]

    fetched_shard = as_shards.fetch_shard("test1")
    assert fetched_shard["packages"]["test10.tar.bz2"]["name"] == "test1"
    assert as_shards.url in repr(as_shards)
    assert "test1" in as_shards

    fetched_shards = as_shards.fetch_shards(("test1", "test2"))
    assert len(fetched_shards) == 2
    assert fetched_shards["test1"]
    assert fetched_shards["test2"]

    as_shards.visited.update(fetched_shards)
    as_shards.visited["package-that-does-not-exist"] = None
    repodata = as_shards.build_repodata()
    assert len(repodata["packages"]) == 3
    assert len(repodata["packages.conda"]) == 3


def test_shardlike_repr():
    """
    Code coverage for ShardLike.__repr__()
    """
    shardlike = ShardLike(
        {
            "packages": {},
            "packages.conda": {},
            "info": {"base_url": "", "shards_base_url": "", "subdir": "noarch"},
        },
        "https://conda.anaconda.org/",
    )
    cls, url, *_ = repr(shardlike).split()
    assert "ShardLike" in cls
    assert shardlike.url == url


def test_shard_hash_as_array():
    """
    Test that shard hashes can be bytes or list[int], for rattler compatibility.
    """
    name = "package"
    fake_shard: ShardsIndexDict = {
        "info": {"subdir": "noarch", "base_url": "", "shards_base_url": ""},
        "repodata_version": 1,
        "shards": {
            name: list(hashlib.sha256().digest()),  # type: ignore
        },
    }

    fake_shard_2 = fake_shard.copy()
    fake_shard_2["shards"] = fake_shard["shards"].copy()
    fake_shard_2["shards"][name] = hashlib.sha256().digest()

    assert isinstance(fake_shard["shards"][name], list)
    assert isinstance(fake_shard_2["shards"][name], bytes)

    index = Shards(fake_shard, "", None)  # type: ignore
    index_2 = Shards(fake_shard_2, "", None)  # type: ignore

    shard_url = index.shard_url(name)
    shard_url_2 = index_2.shard_url(name)
    assert shard_url == shard_url_2


def test_ensure_hex_hash_in_record():
    """
    Test that ensure_hex_hash_in_record() converts bytes to hex strings.
    """
    name = "package"
    sha256_hash = hashlib.sha256()
    md5_hash = hashlib.md5()
    for sha, md5 in [
        (sha256_hash.digest(), md5_hash.digest()),
        (list(sha256_hash.digest()), list(md5_hash.digest())),
        (sha256_hash.hexdigest(), md5_hash.hexdigest()),
    ]:
        record = {
            "name": name,
            "sha256": sha,
            "md5": md5,
        }

        updated = shards.ensure_hex_hash(record)  # type: ignore
        assert isinstance(updated["sha256"], str)  # type: ignore
        assert updated["sha256"] == sha256_hash.hexdigest()  # type: ignore
        assert isinstance(updated["md5"], str)  # type: ignore
        assert updated["md5"] == md5_hash.hexdigest()  # type: ignore


ROOT_PACKAGES = [
    "__archspec",
    "__conda",
    "__osx",
    "__unix",
    "bzip2",
    "ca-certificates",
    "expat",
    "icu",
    "libexpat",
    "libffi",
    "liblzma",
    "libmpdec",
    "libsqlite",
    "libzlib",
    "ncurses",
    "openssl",
    "pip",
    "python",
    "python_abi",
    "readline",
    "tk",
    "twine",
    "tzdata",
    "xz",
    "zlib",
]


def test_build_repodata_subset(prepare_shards_test: None, tmp_path):
    """
    Build repodata subset using the third attempt at a dependency traversal
    algorithm.
    """

    # installed, plus what we want to add (twine)
    root_packages = ROOT_PACKAGES[:]

    channels = list(context.default_channels)
    channels.append(Channel("conda-forge-sharded"))

    with _timer("build_repodata_subset()"):
        channel_data = build_repodata_subset(root_packages, channels)

    # convert to PackageInfo for libmamba, without temporary files
    package_info = []
    for channel, shardlike in channel_data.items():
        repodata = shardlike.build_repodata()
        # Don't like going back and forth between channel objects and URLs;
        # build_repodata_subset() expands channels into per-subdir URLs as
        # part of fetch:
        channel_object = Channel(channel)
        channel_id = str(channel_object)
        for package_group in ("packages", "packages.conda"):
            for filename, record in repodata.get(package_group, {}).items():
                package_info.append(
                    _package_info_from_package_dict(
                        record,
                        filename,
                        url=shardlike.url,
                        channel_id=channel_id,
                    )
                )

    assert len(package_info), "no packages in subset"

    print(f"{len(package_info)} packages in subset")

    with _timer("write_repodata_subset()"):
        repodata_size = repodata_subset_size(channel_data)
    print(f"Repodata subset would be {repodata_size} bytes as json")

    # e.g. this for noarch and osx-arm64
    # % curl https://conda.anaconda.org/conda-forge-sharded/noarch/repodata.json.zst | zstd -d | wc
    full_repodata_benchmark = 138186556 + 142680224

    print(
        f"Versus only noarch and osx-arm64 full repodata: {repodata_size / full_repodata_benchmark:.02f} times as large"
    )

    print("Channels:", ",".join(urllib.parse.urlparse(url).path[1:] for url in channel_data))


def test_shards_indexhelper(prepare_shards_test):
    """
    Load LibMambaIndexHelper with parameters that will enable sharded repodata.

    This will include a build_repodata_subset() call redundant with
    test_build_repodata_subset().
    """
    channels = [Channel("conda-forge-sharded")]

    class fake_in_state:
        installed = {name: object() for name in ROOT_PACKAGES}
        requested = ("vaex",)

    # Would eagerly download repodata.json.zst for all channels
    helper = LibMambaIndexHelper(
        channels,
        (),  # subdirs
        "repodata.json",
        (),
        (),  # pkgs_dirs to load packages locally when offline
        in_state=fake_in_state,  # type: ignore
    )

    print(helper.repos)


def test_batch_retrieve_from_cache(prepare_shards_test: None):
    """
    Test single database query to fetch cached shard URLs in a batch.
    """
    channels = [*context.default_channels, Channel("conda-forge-sharded")]
    roots = [
        Node(distance=0, package="ca-certificates", visited=False),
        Node(distance=0, package="icu", visited=False),
        Node(distance=0, package="expat", visited=False),
        Node(distance=0, package="libexpat", visited=False),
        Node(distance=0, package="libffi", visited=False),
        Node(distance=0, package="libmpdec", visited=False),
        Node(distance=0, package="libzlib", visited=False),
        Node(distance=0, package="openssl", visited=False),
        Node(distance=0, package="python", visited=False),
        Node(distance=0, package="readline", visited=False),
        Node(distance=0, package="liblzma", visited=False),
        Node(distance=0, package="xz", visited=False),
        Node(distance=0, package="libsqlite", visited=False),
        Node(distance=0, package="tk", visited=False),
        Node(distance=0, package="ncurses", visited=False),
        Node(distance=0, package="zlib", visited=False),
        Node(distance=0, package="pip", visited=False),
        Node(distance=0, package="twine", visited=False),
        Node(distance=0, package="python_abi", visited=False),
        Node(distance=0, package="tzdata", visited=False),
    ]

    with _timer("repodata.json/shards index fetch"):
        channel_data = fetch_channels(channels)

    with _timer("Shard fetch"):
        sharded = [channel for channel in channel_data.values() if isinstance(channel, Shards)]
        assert sharded, "No sharded repodata found"
        remaining = batch_retrieve_from_cache(sharded, [node.package for node in roots])
        print(f"{len(remaining)} shards to fetch from network")

    # execute "no sharded channels" branch
    remaining = batch_retrieve_from_cache([], ["python"])
    assert remaining == []

    # XXX don't call everything Shard/Shards


def test_shards_connections(monkeypatch):
    """
    Test _shards_connections() and execute all its code.
    """
    assert context.repodata_threads is None
    assert _shards_connections() == 10  # requests' default

    poolmanager = (
        get_session("https://repo.anaconda.com/pkgs/main").get_adapter("https://").poolmanager
    )  # type: ignore
    monkeypatch.setattr(poolmanager, "connection_pool_kw", {"no_maxsize": 0})

    monkeypatch.setattr(shards, "SHARDS_CONNECTIONS_DEFAULT", 7)
    assert _shards_connections() == 7

    monkeypatch.setattr(context, "_repodata_threads", 4)
    assert _shards_connections() == 4
