# Future work

```{warning} WIP
These will/should be expanded into issues.
```

* `MatchSpec` preparation logic contains too many exceptions for `libmamba` (e.g. differences in `update` vs `install`)
* Clean-up `MatchSpec` exchange fixes and workarounds
* Channel names and URLs need pre- and post- treatment (e.g. URL escaping) to workaround escaping issues, etc.
* Investigate better usage of `libsolv` and `libmamba` flags
* Better condense retry logic and conflict handling; we lack the "Optional" feature `classic` has
* Investigate ignored tests in `pyproject.toml`
* Conversely, investigate which other tests we should ignore in the name of simplifying the logic
