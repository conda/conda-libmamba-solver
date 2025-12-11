# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
Indirection to zstd implementation.
"""

try:
    from compression.zstd import (
        ZstdCompressor,
        ZstdDecompressor,
        ZstdError,
        ZstdFile,
        compress,
        decompress,
    )
except ImportError as e:
    try:
        from backports.zstd import (
            ZstdCompressor,
            ZstdDecompressor,
            ZstdError,
            ZstdFile,
            compress,
            decompress,
        )
    except ImportError:
        raise ImportError(name="backports.zstd") from e

__all__ = (
    "ZstdCompressor",
    "ZstdDecompressor",
    "ZstdError",
    "ZstdFile",
    "compress",
    "decompress",
)
