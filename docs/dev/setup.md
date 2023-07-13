# How to set up your development environment

## With `conda/conda` Docker images

The development workflow is streamlined for Linux thanks to the `conda/conda` Docker images used in the upstream CI.

1. Clone `conda/conda`, `mamba-org/mamba` (optional), `conda/conda-libmamba-solver` to your preferred locations
   (e.g. `~/devel/conda`, `~/devel/mamba` and `~/devel/conda-libmamba-solver`, respectively).
2. Run the `conda/conda` Docker images with the repos mounted to the following locations.
   In this case we are using `amd64` with `python=3.10` by default, but feel free to customize if needed. If you are running on Apple Silicon, you can use the `linux/aarch64` platform instead of `linux/amd64` for faster performance. Note that some tests might fail due to the different architecture. You can choose between `defaults` or `conda-forge` based images:

```bash
# For defaults-based images, use:
$ docker run -it --rm \
    --platform=linux/amd64 \
    -v ~/devel/conda:/opt/conda-src \
    -v ~/devel/mamba:/opt/mamba-src \
    -v ~/devel/conda-libmamba-solver:/opt/conda-libmamba-solver-src \
    ghcr.io/conda/conda-ci:main-linux-python3.10 \
    bash
# For conda-forge-based images, use the following instead:
$ docker run -it --rm \
    --platform=linux/amd64 \
    -v ~/devel/conda:/opt/conda-src \
    -v ~/devel/mamba:/opt/mamba-src \
    -v ~/devel/conda-libmamba-solver:/opt/conda-libmamba-solver-src \
    ghcr.io/conda/conda-ci:main-linux-python3.10-conda-forge \
    bash
```

3. This will drop you in a `bash` session with the `conda`, `mamba`, and `conda-libmamba-solver` repositories
   mounted to `/opt/conda-src`, `/opt/mamba-src`, and `/opt/conda-libmamba-solver-src`, respectively.
   `/opt/conda` contains the bundled Miniconda installation.

4. Set up the development environment with `source /opt/conda-libmamba-solver-src/dev/linux/bashrc.sh`.

5. Enjoy! Since the local repositories are mounted, you can make modifications to the source live,
   and they will be reflected in the Docker instance automatically.
   Run `pytest` or `conda` as needed, no need to reload Docker!
   However, if the debugging exercises result in a permanent modification of the development environment,
   consider exiting Docker (via <kbd>Ctrl</kbd>+<kbd>D</kbd>) and starting step 2 again.

> **Note** Whenever code changes to C++ `libmamba` source it will
> require a manual recompilation running the command `recompile-mamba`
> Rebuild takes around 1-3 minutes

## General workflow

We strongly suggest you start with the Docker-based workflow above.
It is a better development experience with a fully disposable environment.
However, sometimes you might need to debug issues for non-Linux installations.
In that case, you can follow these general instructions,
but be careful with overwriting your existing `conda` installations,
especially when it comes to `shell` initialization!

1. Get yourself familiar with the ["Development environment" guide for `conda` itself][conda_dev].

2. Fork and clone the `conda-libmamba-solver` repository to your preferred location:

```bash
$ git clone "git@github.com:$YOUR_USERNAME/conda-libmamba-solver" "$REPO_LOCATION"
$ cd "$REPO_LOCATION"
```

3. Install the required dependencies for `conda-libmamba-solver`:

```bash
$ conda install \
    --file "$REPO_LOCATION"/dev/requirements.txt \
    --file "$REPO_LOCATION"/tests/requirements.txt
```

4. Install `conda-libmamba-solver` with `pip`:

```bash
$ cd $REPO_LOCATION
$ python -m pip install --no-deps -e .
```

5. Install the test collection plugins (only for upstream tests in `conda/conda`):

```bash
$ cd $REPO_LOCATION
$ python -m pip install dev/collect_upstream_conda_tests/
```

For testing out the `libmamba` solve you can set it several ways:
 - environment variable `CONDA_SOLVER=libmamba`
 - pass a flag `--solver=libmamba`
 - setting the conda config `conda config --set solver=libmamba` which will modify your `condarc`

## Debugging `conda` and `conda-libmamba-solver`

Once you have followed the steps described in the general workflow
above you may need to investigate the state in a particular
point. Insert a
[`breakpoint()`](https://docs.python.org/3/library/pdb.html) within
the code and run a test or conda directly to hit the breakpoint.

## Debugging Mamba

While debugging the conda workflows only requires modifying python
code and running conda. Debugging the mamba code requires
recompilation and is not as easy to jump into a debugger to
investigate state.

1. Get familiar with the ["Local development" guide for `mamba` itself][mamba_dev].

2. Fork and clone the `mamba` repository to your preferred location:

```bash
$ git clone "git@github.com:$YOUR_USERNAME/mamba" "$REPO_LOCATION"
# cd $REPO_LOCATION
```

3. Use the docker image for development suggested above and re-run
   `recompile-mamba` whenever you make change to `mamba` in
   $REPO_LOCATION. This should take less than a minute.

We recommend debugging via either breakpoints and using gdb or print
statements via `std::cout << ... << std::endl`. The
[following](https://github.com/costrouc/mamba/commit/99ac04ee9ca26c9579c67816cfba25bf310c30fb)
shows an example of inserting print statements into the `libmamba`
source in order to debug the [libsolv](https://github.com/openSUSE/libsolv) state.

<!-- LINKS -->

[conda_dev]: https://docs.conda.io/projects/conda/en/latest/dev-guide/development-environment.html
[mamba_dev]: https://mamba.readthedocs.io/en/latest/developer_zone/build_locally.html
