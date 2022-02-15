import os
import sys
from io import UnsupportedOperation
from time import sleep
from threading import Thread


class CapturedDescriptor:
    """
    Class used to grab standard output or another stream.
    """

    def __init__(self, stream=sys.stdout, threaded=False, sentinel="\b"):
        self._captured_stream = stream
        self._threaded = threaded
        self._sentinel = sentinel
        self.text = ""
        if sys.platform.startswith("win"):
            # This method of capturing stderr messes up downloads on libmamba 0.21
            # Disable until we find a solution
            self.start = self._noop
            self.stop = self._noop
        else:
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
        # Close the pipe:
        os.close(self._pipe_in)
        os.close(self._pipe_out)
        # Restore the original stream:
        os.dup2(self._original_stream_fd, self._captured_stream_fd)
        # Close the duplicate stream:
        os.close(self._original_stream_fd)

    stop = _stop

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
