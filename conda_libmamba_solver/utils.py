import os
import sys
from io import UnsupportedOperation
from time import sleep
from threading import Thread
import traceback
import tempfile


class CapturedDescriptor:
    """
    Class used to grab standard output or another stream.
    """

    def __init__(self, stream=sys.stdout, threaded=False, sentinel="\0"):
        self._captured_stream = stream
        self._threaded = threaded
        self._sentinel = sentinel
        self.text = ""
        try:
            # Keep a reference to the descriptor we are capturing
            self._captured_stream_fd = self._captured_stream.fileno()
        except UnsupportedOperation:
            # This happens in our tests because we are already
            # capturing sys.stdout and stderr; if that's the case
            # make this a noop
            self.start = self._noop
            self.stop = self._noop
        else:
            # Create a pipe so the stream can be captured:
            self._pipe_out, self._pipe_in = os.pipe()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def _noop(self, *args, **kwargs):
        pass

    def _start(self):
        """
        Start capturing the stream data.
        """
        self.text = ""
        # Save a copy of the stream:
        self._original_stream_fd = os.dup(self._captured_stream_fd)
        # Replace the original stream with our write pipe:
        os.dup2(self._pipe_in, self._captured_stream_fd)
        if self._threaded:
            # Start thread that will read the stream:
            self._thread = Thread(target=self._read)
            self._thread.start()
            # Make sure that the thread is running and os.read() has executed:
            sleep(0.01)

    start = _start

    def _stop(self):
        """
        Stop capturing the stream data and save the text in `text`.
        """
        try:
            if sys.platform.startswith("win"):
                os.write(self._pipe_in, self._sentinel.encode())
                os.fsync(self._pipe_in)
            else:
                # Print the escape character to make the _read method stop:
                self._captured_stream.write(self._sentinel)
                # Flush the stream to make sure all our data goes in before
                # the escape character:
                self._captured_stream.flush()
            if self._threaded:
                # wait until the thread finishes so we are sure that
                # we have the last character:
                self._thread.join()
            else:
                self._read()
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            self._close()
            raise
        else:
            self._close()

    stop = _stop

    def _close(self):
        # Close the pipe:
        os.close(self._pipe_in)
        os.close(self._pipe_out)
        # Restore the original stream:
        os.dup2(self._original_stream_fd, self._captured_stream_fd)
        # Close the duplicate stream:
        os.close(self._original_stream_fd)

    def _read(self, length=1):
        """
        Read the stream data (one byte at a time)
        and save the text in `text`.
        """
        while True:
            char = os.read(self._pipe_out, length).decode(self._captured_stream.encoding)
            if not char or self._sentinel in char:
                break
            self.text += char


class CaptureStreamToFile:
    def __init__(self, stream=sys.stderr, path=None, callback=None):
        self._original_stream = stream
        if path is None:
            self._file = tempfile.TemporaryFile(mode='w+t')
        else:
            self._file = open(path, mode="a+t")
        self.text = None
        self._callback = callback

    def start(self):
        self._original_fileno = self._original_stream.fileno()
        self._saved_original_fileno = os.dup(self._original_fileno)
        os.dup2(self._file.fileno(), self._original_fileno)

    def stop(self):
        os.dup2(self._saved_original_fileno, self._original_fileno)
        os.close(self._saved_original_fileno)
        self._file.flush()
        self._file.seek(0)
        self.text = self._file.read()
        self._file.close()
        if self._callback:
            self._callback(self.text)

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
