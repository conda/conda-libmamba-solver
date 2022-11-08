# Reasons to use `conda-libmamba-solver`... or not

You should definitely try `conda-libmamba-solver` if:

* You want a faster solver with low-memory footprint
* Some of your environments do not solve quick enough, or at all, with `classic`
* You are okay with slightly different (but metadata-compliant) solutions.

> Users most often find alternative solutions surprising when they request packages with very few restraints. 
> If the given solution is not fully satisfying, try to restrict your request a bit more. 
> For example, if you run `conda install scipy` and do not get the latest version, try using a more explicit command: `conda install scipy=X.Y`. 

You should probably still use `classic` in the following scenarios:

* Backwards compatibility is very important in your use case, and you need your environments solved in the same way they have been solved in the past years. 
  This is a bit more important in long-lived environments than new ones.

> Note that you can always use `--solver=classic` to re-enable the `classic` solver temporarily for specific operations, even after setting `libmamba` as default.

## Differences between `libmamba` and `classic`

Ok, so we know `conda-libmamba-solver` brings a faster solver to `conda`. 
But, why is that? Or, in other words, why couldn't `classic` become faster on its own? 

The following sections provide deeper technical details about the reasons, both at the implementation and algorithmic level.
If you don't care about that much detail, just know that:

* Deep within, both `classic` and `conda-libmamba` rely on C-based code to solve the SAT problem.
  However, `classic` uses Python objects to define and manage the SAT clauses, which incurs a large overhead.
  `libmamba` lets `libsolv` do the heavy lifting, operating in C++ and C, respectively.
  `conda-libmamba-solver` tries to delegate to the `libmamba` and `libsolv` compiled libraries as soon as possible to minimize the Python overhead.
* `classic` has a more involved retry-logic than can incur in more time-consuming solver attempts, especially for existing environments.
* Both options use SAT solvers, but they invoke them differently.
  `classic` uses a multistep, multi-objective optimization scheme, which resembles a global optimization scheme.
  `libsolv` opts for a backtracking alternative, closer to a local optimization scheme.
  This can result in `libmamba` choosing a different member of the whole solution ensemble.

### Implementation differences

Let's first analyze how both solvers are implemented. 

The `classic` solver logic is distributed across several abstraction layers in `conda`.

* `conda.cli.install`: 
  This module contains the base implementation for `conda [env] install|remove|update|create`. 
  It eventually delegates to the `Solver` class, after some preparation tasks. 
  This module can run up to 4 solver attempts by default: use `current_repodata.json` first, or retry with `repodata.json`, and with and without the `--freeze-installed` flag.
* `conda.core.solve.Solver`: 
  This class provides a three-function API that interfaces with the `Transaction` system. 
  Almost of all the logic falls under the `Solver.solve_final_state()` method. 
  At this step, `classic` downloads the channels metadata, collects information about the target environment and applies the command-line instructions provided by the user. 
  The end result is a list of `MatchSpec` objects;
  in other words, a list of constraints that underlying solver must use to best select the needed packages from the channels.
* `conda.resolve.Resolve`: 
  This class receives the `MatchSpec` instructions from the higher level `Solver` class and transforms them into SAT clauses, as implemented in the `conda.common.logic.Clauses` and `conda.common._logic.Clauses` classes. 
  [`Resolve.solve()`][Resolve.solve] is the method that governs the algorithmic details of "solving the environment".
* `conda.common._logic._SatSolver`: 
  Provides the parent class for all three SAT solver wrappers implemented as part of the `classic` logic (PycoSat, PyCryptoSat, PySat). 
  The default one is `PycoSat`, but you can change it with the `sat_solver` option in your configuration.
* `conda.common._logic._PycoSatSolver`: This class wraps the `pycosat` bindings to `picosat`, the underlying C library that actually solves the SAT clauses problem.

For `conda-libmamba-solver`, we initially tried to provide an implementation at the `_SatSolver` level, 
but `libsolv` (and hence `libmamba`) didn't expose a SAT-based API. 
We ended up with an implementation a bit higher up in the abstraction tree:

* `conda.cli.install`: 
  We always ignore `current_repodata.json` and implement the `--freeze-installed` attempts closer to the solver so we don't have to re-run the preparation steps.
* `conda_libmamba_solver.solver.LibmambaSolver`: 
  A `conda.core.solve.Solver` subclass that completely replaces the `Solver.solve_final_state()` method. 
  We used this opportunity to refactor some of the pre-solver logic (spread across different layers in `classic`) into a solver-agnostic module (`conda_libmamba_solver.state`) with nicer-to-work-with helper objects. 
  Our subclass instantiates the `libmamba` objects.
* `libmamba.api.Solver`: 
  The `libmamba.api` Python module is generated by `pybind11` bindings to the underlying `libmamba` C++ library. 
  Some of the objects we rely on are `api.Solver` (interfaces with `libsolv`), `api.Context` (reimplementation of `conda.base.context`) and the `api.{Pool,Repo}` stack (handles the channel metadata and target environment state).
* `libsolv`: 
  `libmamba` relies on this C project directly to handle the solving steps. 
  The conda-specific logic is implemented in the [`conda.c`][conda.c] file.

The implementation details reveal some of the reasons for the performance differences:

* `classic` uses many Python layers before it finally reaches the compiled code (`picosat`): 
    * Tens of `MatchSpec` objects reflect the input state: installed packages, system constraints and user-requested packages
    * The channel index (repodata files) results in tens of thousands of `PackageRecord` objects
    * The SAT clauses end up being expressed as tens or hundreds of thousands of `logic.Clauses` and `_logic.Clauses` objects.
    * The optimization algorithm in `Resolve.solve()` invokes picosat several times, switching between Python and C contexts very often, recreating `Clauses` as necessary.
* `conda-libmamba-solver`, in contrast, switches to C++ pretty early in the abstraction tree.  
    * Only the `MatchSpec` objects are created in the Python layer, only to be immediately forwarded to its C++ counterparts.
    * The SAT clauses are built and handled by `libsolv`, using a [very memory-efficient approach based on 32-bit integers only][libsolv-history]. 
      This allows the SAT problem to [be treated in milliseconds][libsolv-ms].

### Algorithmic details

#### Retry logic

`classic` tries hard to minimally modify your environment, so by default, the flag `--freeze-installed` will be applied. 
This means all your installed packages will be constrained to their current installed version. 
If the SAT solver couldn't find a solution, then `classic` will analyze which packages are causing the conflict. 
If the conflicting packages were not explicitly requested by the user (in the current or previous operations in the target environment), 
their version constraint will be relaxed and a new solving attempt will be made. 

If, despite the progressive constraint relaxation, the SAT solver cannot find a solution, the `Solver` class will raise an exception to the `conda.cli.install` module. 
This will trigger a second round of attempts, without `--freeze-installed`. 
In simplified Python:

```python
for repodata in ("current_repodata.json", "repodata.json"):
    solver = Solver(repodata_fn=repodata)
    for should_freeze in (True, False):
        success = solver.solve(freeze_installed=True)
        if success:
            break
else:
    raise SolverError()

class Solver:
    "Super simplified version. Actual implementation is spread across many layers"
    def solve(...):
        index = download_channel(channels, repodata_fn)
        constraints = collect_metadata(target_environment, user_requested_packages)
        while True:
            sat_solver = SATSolver(index)
            clauses = sat_solver.build_clauses(constraints) # expensive!
            success = sat_solver.solve(clauses) # multi-step optimization
            if success:
                return True
            else:
                conflicts = sat_solver.find_conflicts()
                initial_constraints = constraints.copy()
                constraints.update(conflicts)
                if initial_constraints == constraints:
                    return False

```

A similar retry logic is implemented in `conda_libmamba_solver`, 
but `libsolv` gives us the conflicting packages as part of the solving attempt result for free, which allows us to iterate faster. 
We don't need a separate attempt to disable `--freeze-installed` because our retry logic handles conflicts and frozen packages in the same way. 
Additionally, this retry logic can also be disabled or reduced with an environment variable for extreme cases (very large environments). 
We also ignore `current_repodata.json` altogether. 
All of these changes make the overall logic simpler and faster, which compounds on top of the lightning-fast `libmamba` implementation. 

#### SAT algorithms

Given a set of `MatchSpec` objects, `classic` will apply a multistep, multi-objective optimization strategy that invokes the actual SAT solver several times:

* [`conda.resolve.Resolve.solve()`][Resolve.solve] will optimize several objective metrics. 
  In no particular order, some of these rules are:
    * Maximize the versions and build numbers of required packages
    * Minimize the number of `track_features`
    * Prefer non-`noarch` over `noarch` if both are available
    * Minimize the number of necessary upgrades and/or removals
* `conda.common._logic.Clauses.minimize()`: 
  This is used for each step above, and involves a series of SAT calls per minimization. 
  All of these calls involve, at some point, passing Python objects over to the C context, which incurs some overhead.

In contrast, `libmamba` delegates fully to `libsolv`, which has its own logic for conda-specific problems. 
You can read more about it in the [`mamba-org/mamba` documentation][mamba_libsolv_docs], but the most important part:

* `libsolv` is a backtracking SAT solver, [inspired][libsolv-history-sat] by [minisat][minisat].
  This means that it explores "branches" of a solution until it finds one that satisfies the input constraints. 
  If we understand `classic`'s approach as a global-like optimization strategy, and one could say `libsolv`'s better resembles a local optimization approach. 
* This means that in the presence of several compatible solutions, `libsolv` might choose one that is different to the one proposed by `classic`. 

> Tip: Large conda-forge migrations often rely on multiple coexisting build variants to ease the transition (e.g. `openssl` v1 to v3). 
> This introduces several alternative branches `libsolv` can end up exploring and selecting, perhaps with surprising results. 
> Being more explicit about the requested packages usually helps get obtaining the expected solution; 
> e.g. if you want to install `scipy=1.0` (the latest version), express that explicitly: `conda install scipy=1.0` instead of `conda install scipy`.

#### Index reduction

`classic` prunes the channel metadata (internally referred to as the "index") in every `Resolve.solve()` call. 
This reduces the search space by excluding packages that won't ever be needed by the current set of input constraints. 
Conversely, this performance optimization step can longer and longer the larger the index gets. 

In `libsolv`, pruning is part of the filtering, sorting and selection mechanism that informs the solver (see [`policy.c`][policy.c] and [`selection.c`][selection.c]).
It runs in C, using memory-efficient data structures.

### IO differences

Currently, `conda-libmamba-solver` still relies on `libmamba`'s repodata fetching logic to download the channel metadata.
This logic can download channel repodata files in parallel. 
Some channel protocols (FTP, S3) are fetched with `conda` objects, which are usually downloaded one by one.

In the future, `libmamba` will use a more efficient [`powerloader`][powerloader]-based approach.
`classic` will also introduce faster repodata downloads in the future, thanks to the [JLAP incremental download approach][jlap].

In both cases, `conda-libmamba-solver` will be able to benefit from both!

## More information

If you want to read (even more) about this, please check the following resources:

* ["Deep dive: solvers" guide in the `conda` documentation][deep-dive]
* ["Package resolution" in the `mamba` documentation][mamba-pkg-resolution]
* [Libsolv documentation][libsolv-docs]
* [Explore ways to use other solvers instead of interacting with SAT solver directly][conda-solvers-issue] (`conda` issue)
* [libsolv for conda?][libsolv-issue] (`libsolv` issue)

<!-- LINKS -->

[conda.c]: https://github.com/openSUSE/libsolv/blob/0.7.22/src/conda.c
[policy.c]: https://github.com/openSUSE/libsolv/blob/0.7.22/src/policy.c
[selection.c]: https://github.com/openSUSE/libsolv/blob/0.7.22/src/selection.c
[Resolve.solve]: https://github.com/conda/conda/blob/22.9.0/conda/resolve.py#L1244
[libsolv-history]: https://github.com/openSUSE/libsolv/blob/0.7.22/doc/libsolv-history.txt
[libsolv-history-sat]: https://github.com/openSUSE/libsolv/blob/0.7.22/doc/libsolv-history.txt#using-sat-for-solving
[libsolv-ms]: https://github.com/openSUSE/libsolv/issues/284#issuecomment-428927641
[mamba_libsolv_docs]: https://mamba.readthedocs.io/en/latest/advanced_usage/package_resolution.html
[minisat]: http://minisat.se/
[jlap]: https://github.com/conda/conda/issues/11640
[powerloader]: https://github.com/mamba-org/powerloader
[deep-dive]: https://docs.conda.io/projects/conda/en/stable/dev-guide/deep-dive-solvers.html
[mamba-pkg-resolution]: https://mamba.readthedocs.io/en/latest/advanced_usage/package_resolution.html
[libsolv-docs]: https://github.com/openSUSE/libsolv/tree/master/doc
[libsolv-issue]: https://github.com/openSUSE/libsolv/issues/284
[conda-solvers-issue]: https://github.com/conda/conda/issues/7808#issuecomment-429805392
