# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
Exceptions used in conda-libmamba-solver
"""

from conda.exceptions import UnsatisfiableError


class LibMambaUnsatisfiableError(UnsatisfiableError):
    """An exception to report unsatisfiable dependencies.
    The error message is passed directly as a str.
    """

    def __init__(self, message: str, **kwargs):
        super(UnsatisfiableError, self).__init__(str(message))
