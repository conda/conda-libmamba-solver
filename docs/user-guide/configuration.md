# Configuration

## Basic options

conda-libmamba-solver is a _solver plugin_ and can be configured with the same settings as the classic solver. This is usually done via the `conda config` subcommand. Read the [conda configuration docs](https://docs.conda.io/projects/conda/en/stable/configuration.html) and check for the general solver options there.

## Advanced options

Additionally, conda-libmamba-solver can be further configured via special environment variables.
We do not recomment using these options in production environments. Their behavior might change in the future, or they can be entirely removed without prior notice.

* `CONDA_LIBMAMBA_SOLVER_MAX_ATTEMPTS`: Maximum number of attempts to find a solution. By default, this is set to the number of installed packages in the environment. In commands that involve a large number of changes in a large environment, it can take a bit to relax the constraints enough to find a solution. This option can be used to reduce the number of attempts and "give up" earlier.
* `CONDA_LIBMAMBA_SOLVER_DEBUG_LIBSOLV`: Enable verbose logging from `libsolv`. Only has an effect if combined with `-vvv` in the CLI. Note that this will incur a big performance overhead. Only use when debugging solver issues.
