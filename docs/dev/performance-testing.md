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

The `linux-benchmarks` job in `Tests` runs the suite on Ubuntu 22.04 with Python 3.14.
The separate `Track Benchmarks` workflow uploads results for pull requests and
commits on `main`, feature branches, and release branches. Testbeds use the
operating system, architecture, Python major/minor version, and CPU model recorded
in the benchmark results. This keeps different CPUs assigned to the same GitHub
runner label in separate histories. Pull requests compare against the base
branch's available history on the same testbed.

Maintainers enable uploads by creating the `conda-libmamba-solver` project in Bencher
and setting the `BENCHER_API_KEY` repository secret to its project API key.
Base branch uploads build a separate history for each testbed.

A green Bencher check means no alert was raised.
[Regression detection](https://bencher.dev/docs/explanation/thresholds/) requires
a threshold and enough matching history for each benchmark. A new testbed can
therefore have a green check before regression detection is possible.

Matching CPU models do not eliminate variation from runner load, runner-image
updates, or dependency changes. Investigate these alongside code changes when
interpreting an alert.
