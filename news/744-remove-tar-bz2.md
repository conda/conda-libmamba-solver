### Bug fixes

* Remove `.tar.bz2` with matching `.conda`-format packages during shard
  traversal if `conda` is not in "use_only_tar_bz2" mode; needed as shards
  directly adds individual packages to the solver. (#710)
