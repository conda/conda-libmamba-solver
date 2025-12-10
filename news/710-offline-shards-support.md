### Enhancements

* Add offline mode support for sharded repodata. When offline mode is enabled, the solver will use cached shards even if they are expired, and gracefully fall back to non-sharded repodata if no cache exists. Missing shards in offline mode return empty shards rather than failing. (#710)
