# Performance tips and tricks

`conda-libmamba-solver` is much faster than classic for [many reasons](./libmamba-vs-classic.md),
but there are certain tricks you can use to make it even faster!

These tips apply to both solvers:

- **Explicit is better**.
  Instead of letting the solve do all the work, specify target versions for your packages.
  `conda install python=3.11 numpy` is way better than `conda install python numpy`.
* Use `--strict-channel-priority`.
  Strict channel priority drastically reduces the solver search space when you are mixing channels.
  Make this decision permanent with `conda config --set channel_priority strict`.
* Use `--update-specs`.
  For existing environments, do not attempt to freeze installed packages by default.

## For `conda-libmamba-solver`

* Experimental: `CONDA_LIBMAMBA_SOLVER_MAX_ATTEMPTS=0`.
  Setting this environment variable will disable the retry loop, making it behave more like `mamba`.

## For conda `classic`

The above tips also apply to `classic`, but you can supplement them with:

* `--repodata-fn=repodata.json` to skip using `current_repodata.json`.
* `CONDA_UNSATISFIABLE_HINTS_CHECK_DEPTH=1` won't help solves get any faster, but failures will be reported more quickly.

## References

- [Understanding and Improving Conda's performance](https://www.anaconda.com/blog/understanding-and-improving-condas-performance)
- [How We Made Conda Faster in 4.7](https://www.anaconda.com/blog/how-we-made-conda-faster-4-7)
