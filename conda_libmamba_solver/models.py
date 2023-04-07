# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
import logging
import sys
from collections import defaultdict
from collections.abc import MutableMapping
from enum import Enum
from typing import Any, Hashable, Iterable, Optional, Tuple, Union

log = logging.getLogger(f"conda.{__name__}")


class TrackedMap(MutableMapping):
    """
    Implements a dictionary-like interface with self-logging capabilities.

    Each item in the dictionary can be annotated with a ``reason`` of type ``str``.
    Since a keyword argument is needed, this is only doable via the ``.set()`` method
    (or any of the derivative methods that call it, like ``.update()``). With normal
    ``dict`` assignment (à la ``d[key] = value``), ``reason`` will be None.

    Reasons are kept in a dictionary of lists, so a history of reasons is kept for each
    key present in the dictionary. Reasons for a given ``key`` can be checked with
    ``.reasons_for(key)``.

    Regardless the value of ``reason``, assignments, updates and deletions will be logged
    for easy debugging. It is in principle possible to track where each key came from
    by reading the logs, since the stack level is matched to the originating operation.

    ``.set()`` and ``.update()`` also support an ``overwrite`` boolean option, set to
    True by default. If False, an existing key will _not_ be overwritten with the
    new value.

    Parameters
    ----------
    name
        A short identifier for this tracked map. Useful for logging.
    data
        Initial data for this object. It can be a dictionary, an iterable of key-value
        pairs, or another ``TrackedMap`` instance. If given a ``TrackedMap`` instance,
        its data and reasons will be copied over, instead of wrapped, to avoid recursion.
    reason
        Optionally, a reason on why this object was initialized with such data. Ignored
        if no data is provided.

    Examples
    --------
    >>> TrackedMap("example", data={"key": "value"}, reason="Initialization)
    >>> tm = TrackedMap("example")
    >>> tm.set("key", "value", reason="First value")
    >>> tm.update({"another_key": "another_value"}, reason="Second value")
    >>> tm["third_key"] = "third value"
    >>> tm[key]
        "value"
    >>> tm.reasons_for(key)
        ["First value"]
    >>> tm["third_key"]
        "third value"
    >>> tm.reasons_for("third_key")
        [None]
    """

    def __init__(
        self,
        name: str,
        data: Optional[Union["TrackedMap", Iterable[Iterable], dict]] = None,
        reason: Optional[str] = None,
    ):
        self._name = name
        self._clsname = self.__class__.__name__

        if isinstance(data, TrackedMap):
            self._data = data._data.copy()
            if reason:
                self._reasons = {k: reason for k in self._data}
            else:
                self._reasons = data._reasons.copy()
        else:
            self._data = {}
            self._reasons = defaultdict(list)
            self.update(data or {}, reason=reason)

    def _set(self, key, value, *, reason: Optional[str] = None, overwrite=True, _level=3):
        assert isinstance(key, str), f"{key!r} is not str ({reason})"
        try:
            old = self._data[key]
            old_reason = self._reasons.get(key, [None])[-1]
            msg = (
                f"{self._clsname}:{self._name}[{key!r}] "
                f"(={old!r}, reason={old_reason}) updated to {self._short_repr(value)}"
            )
            write = overwrite
        except KeyError:
            msg = f"{self._clsname}:{self._name}[{key!r}] set to {self._short_repr(value)}"
            write = True

        if write:
            if reason:
                msg += f" (reason={reason})"
                self._reasons[key].append(reason)
            self._data[key] = value
        else:
            msg = (
                f"{self._clsname}:{self._name}[{key!r}] "
                f"(={old!r}, reason={old_reason}) wanted new value {self._short_repr(value)} "
                f"(reason={reason}) but stayed the same due to overwrite=False."
            )

        self._log_debug(msg, stacklevel=_level)

    @property
    def name(self):
        return self._name

    def __repr__(self):
        if not self._data:
            return "{}"
        lines = ["{"]
        for k, v in self._data.items():
            reasons = self._reasons.get(k)
            reasons = f"  # reasons={reasons}" if reasons else ""
            lines.append(f"  {k!r}: {v!r},{reasons}")
        lines.append("}")
        return "\n".join(lines)

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key: Hashable):
        return self._data[key]

    def __setitem__(self, key: Hashable, value: Any):
        self._set(key, value)

    def __delitem__(self, key: Hashable):
        del self._data[key]
        self._reasons.pop(key, None)
        self._log_debug(f"{self._clsname}:{self._name}[{key!r}] was deleted", stacklevel=2)

    def pop(self, key: Hashable, *default: Any, reason: Optional[str] = None) -> Any:
        """
        Remove a key-value pair and return the value. A reason can be provided
        for logging purposes, but it won't be stored in the object.
        """
        value = self._data.pop(key, *default)
        self._reasons.pop(key, *default)
        msg = f"{self._clsname}:{self._name}[{key!r}] (={self._short_repr(value)}) was deleted"
        if reason:
            msg += f" (reason={reason})"
        self._log_debug(msg, stacklevel=2)
        return value

    def popitem(
        self, key: Hashable, *default: Any, reason: Optional[str] = None
    ) -> Tuple[Hashable, Any]:
        """
        Remove and return a key-value pair. A reason can be provided for logging purposes,
        but it won't be stored in the object.
        """
        key, value = self._data.popitem(key)
        self._reasons.pop(key, *default)
        msg = f"{self._clsname}:{self._name}[{key!r}] (={self._short_repr(value)}) was deleted"
        if reason:
            msg += f" (reason={reason})"
        self._log_debug(msg, stacklevel=2)
        return key, value

    def clear(self, reason: Optional[str] = None):
        """
        Remove all entries in the map. A reason can be provided for logging purposes,
        but it won't be stored in the object.
        """
        self._data.clear()
        self._reasons.clear()
        msg = f"{self._name} was cleared"
        if reason:
            msg += f" (reason={reason})"
        self._log_debug(msg, stacklevel=2)

    def update(
        self, data: Union[dict, Iterable], *, reason: Optional[str] = None, overwrite: bool = True
    ):
        """
        Update the dictionary with a reason. Note that keyword arguments
        are not supported in this specific implementation, so you can only
        update a dictionary with another dictionary or iterable as a
        positional argument. This is done so `reason` and `overwrite` can
        be used to control options instead of silently ignoring a potential
        entry in a ``**kwargs`` argument.
        """
        if hasattr(data, "keys"):
            for k in data.keys():
                self._set(k, data[k], reason=reason, overwrite=overwrite)
        else:
            for k, v in data:
                self._set(k, v, reason=reason, overwrite=overwrite)

    def set(
        self, key: Hashable, value: Any, *, reason: Optional[str] = None, overwrite: bool = True
    ):
        """
        Set ``key`` to ``value``, optionally providing a ``reason`` why.

        Parameters
        ----------
        key
            Key to the passed value
        value
            Value
        reason
            A short description on why this key, value pair was added
        overwrite
            If False, do _not_ update the ``value`` for ``key`` if ``key``
            was already present in the dictionary.
        """
        self._set(key, value, reason=reason, overwrite=overwrite)

    def reasons_for(self, key: Hashable) -> Union[Iterable[Union[str, None]], None]:
        """
        Return the stored reasons for a given ``key``
        """
        return self._reasons.get(key)

    def copy(self):
        return self.__class__(name=self._name, data=self)

    @staticmethod
    def _short_repr(value, maxlen=100):
        value_repr = repr(value)
        if len(value_repr) > maxlen:
            value_repr = f"{value_repr[:maxlen-4]}...>"
        return value_repr

    def _log_debug(self, *args, **kwargs):
        # stacklevel was only added to logging in py38
        if sys.version_info < (3, 8):
            kwargs.pop("stacklevel", None)
        log.debug(*args, **kwargs)


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

    def __getattr__(self, name: str) -> Any:
        if name in ("name", "value"):
            return getattr(self._enum, name)
        if name in self._names:
            return self._enum.name == name
        raise AttributeError(f"'{name}' is not a valid name for {self._enum.__class__.__name__}")

    def __eq__(self, obj: object) -> bool:
        return self._enum.__eq__(obj)

    def _dict(self):
        return {name: self._enum.name == name for name in self._names}
