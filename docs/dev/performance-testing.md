# Performance testing

Performance benchmarks use [pytest-benchmark](https://pytest-benchmark.readthedocs.io/)
and are tracked in [Bencher](https://bencher.dev/).

## Running benchmarks locally

Follow the [development environment setup](setup.md), activate that environment,
and run this command from the repository root:

```shell
python -m pytest tests/test_benchmarks.py -m benchmark --benchmark-only --benchmark-json benchmark_results.json
```

The command writes the same JSON format uploaded by CI. Local runs do not require
Bencher credentials. Benchmarks are marked with `@pytest.mark.benchmark` and use
the `benchmark` fixture to time the target operation.

The suite measures `LibMambaIndexHelper` construction from 100 and 1,000 installed
`PackageRecord` objects and `solve_final_state` against the local channel in
`tests/data/mamba_repo`. Benchmarks run offline for five rounds, with setup outside
the measured work.

## Benchmark tracking in CI

The `linux-benchmarks` job in `Tests` runs the suite on Ubuntu 24.04 with Python 3.14.
For pull requests, it measures the exact base and head revisions sequentially on
the same runner with the same resolved dependencies and `PYTHONHASHSEED=0`.
This roughly doubles benchmark execution time. Both revisions run the PR head's
benchmark cases, fixtures, and test data. The base production code stays at its
original commit. Its benchmark JSON can therefore mark the checkout as dirty
because the tests have been overlaid. `runner_metadata.json` records the head
harness revision separately from the measured source commit and includes the
runner image version. Rename a benchmark when changing the timed operation or
its fixtures so that unlike measurements do not share a long-term history.

The separate `Track Benchmarks` workflow uses the shared Bencher reporting action
in [`conda/actions`](https://github.com/conda/actions) with the Bencher project API
key. The action checks the measured commit IDs, harness revision, runner metadata,
and matching benchmark names. Following Bencher's
[relative benchmarking example](https://bencher.dev/docs/how-to/track-benchmarks/#relative-continuous-benchmarking),
the initial alert tolerance is a 25% latency increase against that job's base
measurement. This is an initial choice to tune with observed noise, not a
statistical confidence level. New benchmark names have no comparison until the
base implementation can run them. If the base run fails or the benchmark names
differ, CI reports a neutral `Benchmark comparison` check. The head results
remain available in the `benchmark-results-v3` artifact alongside the workflow
event and runner diagnostics.

Each PR run stores its base measurements in a separate Bencher branch named
`pr-N-base-RUNID-ATTEMPT`. The `pr-N` comparison starts from that run's base results.
PR baseline uploads never reset or add measurements to the history for `main`,
feature branches, or release branches.

Non-PR uploads build those histories separately for each testbed. Testbeds use
`ubuntu-24.04`, architecture, Python major/minor version, and the CPU model from
the benchmark results. This separates different CPUs assigned to the same GitHub
runner label and keeps Ubuntu 22.04 measurements out of the new histories.
Historical alerts use a t-test with a 0.99 prediction level, at least 10 prior
samples, and at most 64. A green Bencher check means no alert was raised. A new
testbed can have a green check before enough matching history exists for
[regression detection](https://bencher.dev/docs/explanation/thresholds/).

Matching CPU models and using the same runner reduce variation. Runner load,
cache state, image updates, and dependency changes still need investigation when
interpreting an alert. Both PR revisions use the head revision's dependency
resolution, so this comparison measures source changes in that environment.
The artifact includes the dependency list and a best-effort `bencher noise`
diagnostic recorded before benchmarks. Noise diagnostics neither change the
measurements nor determine whether the workflow passes.

Maintainers enable uploads by creating the `conda-libmamba-solver` project in Bencher
and setting the `BENCHER_API_KEY` repository secret to its project API key.
