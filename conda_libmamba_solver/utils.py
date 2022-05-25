import os
import sys
import traceback
import tempfile
from typing import Callable, Optional
from io import UnsupportedOperation
from logging import getLogger
from urllib.parse import quote

from conda import CondaError
from conda.common.compat import on_win
from conda.common.url import urlparse


log = getLogger(f"conda.{__name__}")


class CaptureStreamToFile:
    def __init__(
        self,
        stream=sys.stderr,
        path: Optional[str] = None,
        callback: Optional[Callable] = None,
        keep: Optional[bool] = False,
    ):
        """
        Capture a stream and forward to a file.

        Parameters
        ----------
        stream:
            Which stream to capture. Usually `sys.stderr`.
        path:
            Optional. Path to file in disk. It will be opened with `a+t`. If not
            provided, a temporary file will be used.
        callback:
            A callable that will process the captured text output. It must
            take a single argument of type `str`.
        keep:
            Whether to keep the file contents stored in the `text` attribute.
        """
        self._original_stream = stream
        if path is None:
            self._file = tempfile.TemporaryFile(mode="w+t")
        else:
            self._file = open(path, mode="a+t")
        self.text = None
        self._callback = callback
        self._keep_captured = keep
        self._started = False

    def start(self):
        self._started = False
        self._original_fileno = self._original_stream.fileno()
        self._saved_original_fileno = os.dup(self._original_fileno)
        os.dup2(self._file.fileno(), self._original_fileno)
        self._started = True

    def stop(self):
        if not self._started:
            return
        os.dup2(self._saved_original_fileno, self._original_fileno)
        os.close(self._saved_original_fileno)
        try:
            self._file.flush()
            self._file.seek(0)
            if self._keep_captured:
                self.text = self._file.read()
            if self._callback:
                text = self.text if self._keep_captured else self._file.read()
                self._callback(text)
        finally:
            self._file.close()

    def __enter__(self):
        try:
            self.start()
            return self
        except UnsupportedOperation:
            log.warning("Cannot capture stream! Bypassing ...", exc_info=True)
        except Exception as exc:
            # If there's an exception, we might never see the traceback
            # because STDERR has been captured already. Workaround: print it
            # manually to STDOUT. Note we only do this if the exception is
            # not part of the CondaError family - these exceptions are designed
            # to never print the traceback!
            if not isinstance(exc, CondaError):
                traceback.print_exception(type(exc), exc, None, file=sys.stdout)
            raise exc

    def __exit__(self, exc_type, exc_value, tb):
        try:
            self.stop()
        finally:
            # If there's an exception, we might never see the traceback
            # because STDERR has not been released yet. Workaround: print it
            # manually to STDOUT. Note we only do this if the exception is
            # not part of the CondaError family - these exceptions are designed
            # to never print the traceback!
            if exc_type is not None:
                if not isinstance(exc_value, CondaError):
                    traceback.print_exception(exc_type, exc_value, tb, file=sys.stdout)
                raise exc_value


def escape_channel_url(channel):
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
