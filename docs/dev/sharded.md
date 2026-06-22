# Sharded repodata

`conda-libmamba-solver` supports [CEP-16 Sharded
Repodata](https://conda.org/learn/ceps/cep-0016).

Sharded repodata splits `repodata.json` into an index mapping package names to
shard hashes in `repodata_shards.msgpack.zst`. A shard contains repodata for
every package with a given name. Sharded repodata makes it possible to quickly
build a subset of repodata including all requested packages and their transitive
dependencies; which is much smaller than the repodata for all packages in the
channel.

As of conda-libmamba-solver 26.07, conda-libmamba-solver delegates to conda for
its sharded repodata implementation. `LibMambaIndexHelper` accepts an optional
`build_repodata_subset` function that it uses to fetch sharded repodata. See
[conda's developer
guide](https://docs.conda.io/projects/conda/en/stable/dev-guide/index.html) for
more information.