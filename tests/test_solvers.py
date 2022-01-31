# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from conda.testing.solver_helpers import SolverTests

from conda_libmamba_solver import LibMambaSolver, LibMambaSolverDraft


class TestLibMambaSolver(SolverTests):
    @property
    def solver_class(self):
        return LibMambaSolver

    @property
    def tests_to_skip(self):
        return {
            'LibMambaSolver does not support track-features/features': [
                'test_iopro_mkl',
                'test_iopro_nomkl',
                'test_mkl',
                'test_accelerate',
                'test_scipy_mkl',
                'test_pseudo_boolean',
                'test_no_features',
                'test_surplus_features_1',
                'test_surplus_features_2',
                # this one below only fails reliably on windows;
                # it passes Linux on CI, but not locally?
                'test_unintentional_feature_downgrade',
            ],
            'LibMambaSolver installs numpy with mkl while we were expecting no-mkl numpy': [
                'test_remove',
            ],
        }


class TestLibMambaSolverDraft(TestLibMambaSolver):
    @property
    def solver_class(self):
        return LibMambaSolverDraft
