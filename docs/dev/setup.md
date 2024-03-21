# How to set up your development environment

## With `devcontainer` in VS Code

The development workflow is streamlined for Linux thanks to the `devcontainer` configuration
bundled in this repository. You'll need Docker and VS Code with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers):

1. Clone `conda/conda` and `conda/conda-libmamba-solver` to your preferred locations
   (e.g. `~/devel/conda` and `~/devel/conda-libmamba-solver`, respectively).
   The location does not matter as long as both repositories have the same parent directory.
2. Open your `conda-libmamba-solver` clone with VS Code.
3. Connect to the DevContainer image via the bottom-left menu (<kbd>❱❰</kbd>) and
   click on "Reopen in Container". Pick one of the suggested configurations:
   conda-forge or defaults. The only difference is the base installation (Miniforge and Miniconda,
   respectively).
4. The image will be built and after a couple minutes, you'll be dropped into a Bash shell. Enjoy!
   Since the local repositories are mounted, you can make modifications to the source live,
   and they will be reflected in the Docker instance automatically.
   Run `pytest` or `conda` as needed, no need to reload Docker!
5. If the development environment breaks, click again on <kbd>❱❰</kbd> and, this time, choose
   "Rebuild container". You might need to Retry a couple times.

```{note} Developing libmamba
The devcontainer configuration also supports libmamba 1.x development. You just need to have the
`mamba-org/mamba` repository (branch `1.x`) cloned next to `conda` and `conda-libmamba-solver`.
Once the container has started, run `develop-mamba` to set it up. 
If you are modifying C++ sources, re-run `develop-mamba` to rebuild the libraries.
```

## With regular Docker

You can reuse the devcontainer scripts with regular Docker too.

1. Clone `conda/conda` and `conda/conda-libmamba-solver` to your preferred locations
   (e.g. `~/devel/conda` and `~/devel/conda-libmamba-solver`, respectively).
   You should also clone `mamba-org/mamba` if you need to develop `libmamba`.
   The location does not matter as long as all repositories have the same parent directory.
2. Start a new Docker instance with this command. Adjust the local mounts as necessary.
   
   ```bash
   # For defaults-based images, use:
   $ docker run -it --rm \
      -v ~/devel/conda:/workspaces/conda \
      -v ~/devel/mamba:/workspaces/mamba \
      -v ~/devel/conda-libmamba-solver:/workspaces/conda-libmamba-solver \
      continuumio/miniconda3:latest \
      bash
   # For conda-forge-based images, use the following instead:
   $ docker run -it --rm \
      -v ~/devel/conda:/workspaces/conda \
      -v ~/devel/mamba:/workspaces/mamba \
      -v ~/devel/conda-libmamba-solver:/workspaces/conda-libmamba-solver \
      condaforge/miniforge3:latest \
      bash
   ```
3. Run the `post_create` and `post_start` scripts:
   ```bash
   $ bash /workspaces/conda-libmamba-solver/.devcontainer/post_create.sh
   $ bash /workspaces/conda-libmamba-solver/.devcontainer/post_start.sh
   ```
4. If you want to develop with mamba in editable mode, run:
   ```bash
   $ source ~/.bashrc
   $ develop-mamba
   ```

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
$ cd $REPO_LOCATION
```

3. Use the Docker image for development suggested above and re-run
   `develop-mamba` whenever you make change to `mamba` in
   `$REPO_LOCATION`. This should take less than a minute.

We recommend debugging via either breakpoints and using `gdb` or `print`
statements via `std::cout << ... << std::endl`. The
[following](https://github.com/costrouc/mamba/commit/99ac04ee9ca26c9579c67816cfba25bf310c30fb)
shows an example of inserting print statements into the `libmamba`
source in order to debug the [libsolv](https://github.com/openSUSE/libsolv) state.

<!-- LINKS -->

[conda_dev]: https://docs.conda.io/projects/conda/en/latest/dev-guide/development-environment.html
[mamba_dev]: https://mamba.readthedocs.io/en/latest/developer_zone/build_locally.html
