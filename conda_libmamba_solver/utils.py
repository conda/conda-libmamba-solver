import os
import sys
import traceback
import tempfile
from typing import Callable, Optional
from copy import copy


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

    def start(self):
        self._original_fileno = self._original_stream.fileno()
        self._saved_original_fileno = os.dup(self._original_fileno)
        os.dup2(self._file.fileno(), self._original_fileno)

    def stop(self):
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
        except Exception:
            traceback.print_exception(file=sys.stdout)
            raise

    def __exit__(self, exc_type, exc_value, tb):
        try:
            self.stop()
        finally:
            if exc_type is not None:
                traceback.print_exception(exc_type, exc_value, tb, file=sys.stdout)
                raise exc_type(exc_value)


def safe_conda_build_form(match_spec: "conda.models.match_spec.MatchSpec"):
    """Safe workaround for https://github.com/conda/conda/issues/11347"""
    kwargs = {"name": match_spec.get_exact_value("name")}
    version = match_spec.get_raw_value("version")
    build = match_spec.get_raw_value("build")

    if build:
        kwargs["build"] = build
        kwargs["version"] = version or "*"  # this is the key fix
    elif version:
        kwargs["version"] = version

    return type(match_spec)(**kwargs).conda_build_form()
