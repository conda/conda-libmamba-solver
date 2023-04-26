# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
import sys

from conda.exceptions import UnsatisfiableError


class LibMambaUnsatisfiableError(UnsatisfiableError):
    """An exception to report unsatisfiable dependencies.
    The error message is passed directly as a str.
    """

    def __init__(self, message, **kwargs):
        super(UnsatisfiableError, self).__init__(str(message))


if "conda_build" in sys.modules:
    # I know, gross, but we only need this when conda-build is calling us
    # so we check whether it's already imported, which means it should be
    # safe to import it here.
    from conda_build.exceptions import DependencyNeedsBuildingError

    class ExplainedDependencyNeedsBuildingError(DependencyNeedsBuildingError):
        """
        We need to subclass this to add the explanation to the error message.
        We also add a couple of attributes to make it easier to set up.
        """

        def __init__(self, matchspecs=None, explanation=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.matchspecs = self.matchspecs or matchspecs or []
            self.explanation = explanation

        def __str__(self) -> str:
            msg = self.message
            if not self.explanation:
                # print simple message in log.warn() calls
                return msg
            return "\n".join([msg, self.explanation])

else:
    ExplainedDependencyNeedsBuildingError = LibMambaUnsatisfiableError
