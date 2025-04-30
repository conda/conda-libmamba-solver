# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
This exception is only used in conda-build, so we can't import it directly.
conda_build is not a dependency, but we only import this when conda-build is calling the
solver, so it's fine to import it here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from conda.models.match_spec import MatchSpec

from conda_build.exceptions import DependencyNeedsBuildingError


class ExplainedDependencyNeedsBuildingError(DependencyNeedsBuildingError):
    """
    We need to subclass this to add the explanation to the error message.
    We also add a couple of attributes to make it easier to set up.
    """

    def __init__(
        self,
        matchspecs: Iterable[MatchSpec] | None = None,
        explanation: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.matchspecs = self.matchspecs or matchspecs or []
        self.explanation = explanation

    def __str__(self) -> str:
        msg = self.message
        if not self.explanation:
            # print simple message in log.warning() calls
            return msg
        return "\n".join([msg, self.explanation])
