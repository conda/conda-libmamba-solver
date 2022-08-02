from conda.exceptions import UnsatisfiableError


class LibMambaUnsatisfiableError(UnsatisfiableError):
    """An exception to report unsatisfiable dependencies.
    The error message is passed directly as a str.
    """

    def __init__(self, message, **kwargs):
        super(UnsatisfiableError, self).__init__(str(message))
