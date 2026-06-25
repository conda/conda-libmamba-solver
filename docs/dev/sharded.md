# Sharded repodata

`conda-libmamba-solver` supports [CEP-16 Sharded
Repodata](https://conda.org/learn/ceps/cep-0016).

Sharded repodata splits `repodata.json` into an index mapping package names to
shard hashes in `repodata_shards.msgpack.zst`. A shard contains repodata for
every package with a given name. Sharded repodata makes it possible to quickly
build a subset of repodata including all requested packages and their transitive
dependencies, which is much smaller than the repodata for all packages in the
channel.

As of conda-libmamba-solver 26.6, conda-libmamba-solver delegates to conda for
its sharded repodata implementation. `LibMambaIndexHelper` accepts an optional
`build_repodata_subset` function that it uses to fetch sharded repodata. See
conda's [`build_repodata_subset`
API](https://docs.conda.io/projects/conda/en/stable/dev-guide/api/conda/_private/shards/subset/index.html)
and [`BuildRepodataSubset`
protocol](https://docs.conda.io/projects/conda/en/stable/dev-guide/api/conda/gateways/shards/index.html)
for more information.
