# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
import os
import sys
from textwrap import dedent

from conda.common.io import dashlist
from conda.exceptions import (
    CondaError,
    SpecsConfigurationConflictError,
    UnsatisfiableError,
)


class LibMambaUnsatisfiableError(UnsatisfiableError):
    """An exception to report unsatisfiable dependencies.
    The error message is passed directly as a str.
    """

    def __init__(self, message, **kwargs):
        super(UnsatisfiableError, self).__init__(str(message))


class RequestedAndPinnedError(SpecsConfigurationConflictError):
    """
    Raised when a spec is both requested and pinned.
    """

    def __init__(self, requested_specs, pinned_specs, prefix):
        message = (
            dedent(
                """
                Requested specs overlap with pinned specs.
                  requested specs: {requested_specs_formatted}
                  pinned specs: {pinned_specs_formatted}
                
                Consider adjusting your requested specs to respect the pin(s),
                or explicitly remove the offending pin(s) from the configuration.
                Use 'conda config --show-sources' to look for 'pinned_specs'.
                Pinned specs may also be defined in the file
                {pinned_specs_path}.
                """
            )
            .strip()
            .format(
                requested_specs_formatted=dashlist(requested_specs, 4),
                pinned_specs_formatted=dashlist(pinned_specs, 4),
                pinned_specs_path=os.path.join(prefix, "conda-meta", "pinned"),
            )
        )
        # skip SpecsConfigurationConflictError.__init__ but subclass from it
        # to benefit from the try/except logic in the CLI layer
        CondaError.__init__(
            self,
            message,
            requested_specs=requested_specs,
            pinned_specs=pinned_specs,
            prefix=prefix,
        )


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
