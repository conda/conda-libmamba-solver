# noqa
from conda import CondaError
from conda.auxlib import NULL
from conda.auxlib.ish import dals
from conda.base.constants import (
    REPODATA_FN,
    UNKNOWN_CHANNEL,
    ChannelPriority,
    DepsModifier,
    UpdateModifier,
)
from conda.base.context import context
from conda.common.compat import on_win
from conda.common.io import (
    DummyExecutor,
    Spinner,
    ThreadLimitedThreadPoolExecutor,
    dashlist,
)
from conda.common.path import get_major_minor_version, paths_equal
from conda.common.serialize import json_dump, json_load
from conda.common.url import (
    join_url,
    percent_decode,
    remove_auth,
    split_anaconda_token,
    urlparse,
)
from conda.core.index import _supplement_index_with_system
from conda.core.prefix_data import PrefixData
from conda.core.solve import Solver, get_pinned_specs
from conda.core.subdir_data import SubdirData
from conda.exceptions import (
    InvalidMatchSpec,
    PackagesNotFoundError,
    SpecsConfigurationConflictError,
    UnsatisfiableError,
)
from conda.history import History
from conda.models.channel import Channel
from conda.models.match_spec import MatchSpec
from conda.models.prefix_graph import PrefixGraph
from conda.models.records import PackageRecord
from conda.models.version import VersionOrder
