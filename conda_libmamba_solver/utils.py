import os
import sys
import traceback
import tempfile
from typing import Callable, Optional
from io import UnsupportedOperation
from logging import getLogger


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
            traceback.print_exception(type(exc), exc, None, file=sys.stdout)
            raise

    def __exit__(self, exc_type, exc_value, tb):
        try:
            self.stop()
        finally:
            if exc_type is not None:
                traceback.print_exception(exc_type, exc_value, tb, file=sys.stdout)
                raise exc_type(exc_value)
