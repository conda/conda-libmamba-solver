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
commits on `main`, feature branches, and release branches. Results are grouped by
operating system, architecture, Python version, and CPU model in separate testbeds.
Pull requests are compared with their base branch on the same testbed.

Maintainers enable uploads by creating the `conda-libmamba-solver` project in Bencher
and setting the `BENCHER_API_KEY` repository secret to its project API key.
The first successful upload from `main` establishes the baseline
for each testbed.
