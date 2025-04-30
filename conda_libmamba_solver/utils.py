# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
Miscellaneous utilities
"""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING
from urllib.parse import quote

from conda.common.compat import on_win
from conda.common.url import urlparse
from conda.exceptions import PackagesNotFoundError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from enum import Enum

    from conda.models.match_spec import MatchSpec

    from .index import LibMambaIndexHelper

log = getLogger(f"conda.{__name__}")


def escape_channel_url(channel: str) -> str:
    if channel.startswith("file:"):
        if "%" in channel:  # it's escaped already
            return channel
        if on_win:
            channel = channel.replace("\\", "/")
    parts = urlparse(channel)
    if parts.scheme:
        components = parts.path.split("/")
        if on_win:
            if parts.netloc and len(parts.netloc) == 2 and parts.netloc[1] == ":":
                # with absolute paths (e.g. C:/something), C:, D:, etc might get parsed as netloc
                path = "/".join([parts.netloc] + [quote(p) for p in components])
                parts = parts.replace(netloc="")
            else:
                path = "/".join(components[:2] + [quote(p) for p in components[2:]])
        else:
            path = "/".join([quote(p) for p in components])
        parts = parts.replace(path=path)
        return str(parts)
    return channel


def compatible_specs(
    index: LibMambaIndexHelper, specs: Iterable[MatchSpec], raise_not_found: bool = True
) -> bool:
    """
    Assess whether the given specs are compatible with each other.
    This is done by querying the index for each spec and taking the
    intersection of the results. If the intersection is empty, the
    specs are incompatible.

    If raise_not_found is True, a PackagesNotFoundError will be raised
    when one of the specs is not found. Otherwise, False will be returned
    because the intersection will be empty.
    """
    if not len(specs) >= 2:
        raise ValueError("Must specify at least two specs")

    matched = None
    for spec in specs:
        results = set(index.search(spec))
        if not results:
            if raise_not_found:
                exc = PackagesNotFoundError([spec], index.channels)
                exc.allow_retry = False
                raise exc
            return False
        if matched is None:
            # First spec, just set matched to the results
            matched = results
            continue
        # Take the intersection of the results
        matched &= results
        if not matched:
            return False

    return bool(matched)


class EnumAsBools:
    """
    Allows an Enum to be bool-evaluated with attribute access.

    >>> update_modifier = UpdateModifier("update_deps")
    >>> update_modifier_as_bools = EnumAsBools(update_modifier)
    >>> update_modifier == UpdateModifier.UPDATE_DEPS  # from this
        True
    >>> update_modidier_as_bools.UPDATE_DEPS  # to this
        True
    >>> update_modifier_as_bools.UPDATE_ALL
        False
    """

    def __init__(self, enum: Enum):
        self._enum = enum
        self._names = {v.name for v in self._enum.__class__.__members__.values()}

    def __getattr__(self, name: str) -> bool:
        if name in ("name", "value"):
            return getattr(self._enum, name)
        if name in self._names:
            return self._enum.name == name
        raise AttributeError(f"'{name}' is not a valid name for {self._enum.__class__.__name__}")

    def __eq__(self, obj: object) -> bool:
        return self._enum.__eq__(obj)

    def _dict(self) -> dict[str, bool]:
        return {name: self._enum.name == name for name in self._names}
