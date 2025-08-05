import os
import pickle
from pathlib import Path

from conda_libmamba_solver.index import LibMambaIndexHelper

from conda.base.context import reset_context
from conda.core.subdir_data import SubdirData
from conda.models.channel import Channel

HERE = Path(__file__).parent


def traverse_shards(channels, matchspecs, root_packages):
    assert len(channels) == 1
    channel = next(channels)
    sd = SubdirData(channel)
    fetch = sd.repo_fetch
    cache = fetch.repo_cache
    cache.load_state()  # without reading repodata.json or repodata_shards.msgpack.zst
    print(cache)


def test_shards():
    """
    Reduce full index to only packages needed.
    """
    os.environ['CONDA_TOKEN'] = ""  # test channel doesn't understand tokens
    reset_context()

    in_state = pickle.loads((HERE / "data" / "in_state.pickle").read_bytes())
    print(in_state)

    channels = [Channel.from_url("https://conda.anaconda.org/conda-forge-sharded")]

    installed = (
        *in_state.installed.values(),
        *in_state.virtual.values(),
    )

    helper = LibMambaIndexHelper(
        channels,
        (),  # subdirs
        "repodata.json",
        installed,
        (),  # pkgs_dirs to load packages locally when offline
        in_state=in_state,
    )

    print(helper.repos)

    traverse_shards(channels, in_state.requested, installed)

    """
    >>> in_state.requested
    mappingproxy({'ipython': MatchSpec("ipython")})
    >>> for x in in_state.installed.values(): print(x)
    conda-forge/osx-arm64::bzip2==1.0.8=h99b78c6_7
    conda-forge/noarch::ca-certificates==2025.7.14=hbd8a1cb_0
    conda-forge/osx-arm64::icu==75.1=hfee45f7_0
    conda-forge/osx-arm64::libexpat==2.7.1=hec049ff_0
    conda-forge/osx-arm64::libffi==3.4.6=h1da3d7d_1
    conda-forge/osx-arm64::liblzma==5.8.1=h39f12f2_2
    conda-forge/osx-arm64::libmpdec==4.0.0=h5505292_0
    conda-forge/osx-arm64::libsqlite==3.50.3=h4237e3c_1
    conda-forge/osx-arm64::libzlib==1.3.1=h8359307_2
    conda-forge/osx-arm64::ncurses==6.5=h5e97a16_3
    conda-forge/osx-arm64::openssl==3.5.1=h81ee809_0
    conda-forge/noarch::pip==25.1.1=pyh145f28c_0
    conda-forge/osx-arm64::python==3.13.5=hf3f3da0_102_cp313
    conda-forge/noarch::python_abi==3.13=8_cp313
    conda-forge/osx-arm64::readline==8.2=h1d1bf99_2
    conda-forge/osx-arm64::tk==8.6.13=h892fb3f_2
    conda-forge/noarch::tzdata==2025b=h78e105d_0
    """
