# How to set up your development environment

## With `conda/conda` Docker images

The development workflow is streamlined for Linux thanks to the `conda/conda` Docker images used in the upstream CI.

1. Clone `conda/conda` and `conda/conda-libmamba-solver` to your preferred locations
   (e.g. `~/devel/conda` and `~/devel/conda-libmamba-solver`, respectively).
2. Run the `conda/conda` Docker images with the repos mounted to the following locations.
   In this case we are using `amd64` with `python=3.9` by default, but feel free to customize if needed:

```bash
$ docker run -it --rm \
    --platform=linux/amd64 \
    -v ~/devel/conda:/opt/conda-src \
    -v ~/devel/conda-libmamba-solver:/opt/conda-libmamba-solver-src \
    ghcr.io/conda/conda-ci:main-linux-python3.9 \
    bash
```

3. This will drop you in a `bash` session with the `conda` and `conda-libmamba-solver` repositories
   mounted to `/opt/conda-src` and `/opt/conda-libmamba-solver-src`, respectively.
   `/opt/conda` contains the bundled Miniconda installation.

4. Set up the development environment with `source /opt/conda-libmamba-solver-src/dev/bashrc_linux.sh`.

5. Enjoy! Since the local repositories are mounted, you can make modifications to the source live,
   and they will be reflected in the Docker instance automatically.
   Run `pytest` or `conda` as needed, no need to reload Docker!
   However, if the debugging exercises result in a permanent modification of the development environment,
   consider exiting Docker and starting step 2 again.


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
$ conda install "flit-core>=3.2,<4" \
    --file "$REPO_LOCATION"/dev/requirements.txt \
    --file "$REPO_LOCATION"/tests/requirements.txt
$ conda install -c conda-canary/label/dev conda
```

4. Install `conda-libmamba-solver` with `flit`:

```bash
$ cd $REPO_LOCATION
$ FLIT_ROOT_INSTALL=1 python -m flit install --symlink --deps=none
```

<!-- LINKS -->

[conda_dev]: https://docs.conda.io/projects/conda/en/latest/dev-guide/development-environment.html
