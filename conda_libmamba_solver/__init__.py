# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
try:
    from ._version import version as __version__
except ImportError:
    try:
        from importlib.metadata import version

        __version__ = version("conda_libmamba_solver")
        del version
    except ImportError:
        __version__ = "0.0.0.unknown"


def get_solver_class(key: str = "libmamba"):
    if key == "libmamba":
        from .solver import LibMambaSolver

        return LibMambaSolver
    raise ValueError("Key must be 'libmamba'")
