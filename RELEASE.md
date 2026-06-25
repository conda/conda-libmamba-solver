<!-- edit this in https://github.com/conda/infrastructure -->

[compare]: https://github.com/conda/conda-libmamba-solver/compare
[new release]: https://github.com/conda/conda-libmamba-solver/releases/new
[release docs]: https://docs.github.com/en/repositories/releasing-projects-on-github/automatically-generated-release-notes
[rever docs]: https://regro.github.io/rever-docs
[Anaconda Recipes]: https://github.com/AnacondaRecipes/conda-libmamba-solver-feedstock
[conda-forge]: https://github.com/conda-forge/conda-libmamba-solver-feedstock
[zulip]: https://conda.zulipchat.com/#narrow/channel/480811-releases

# Release Process

> [!NOTE]
> Throughout this document are references to the version number as `YY.MM.MICRO`, this should be replaced with the correct version number. Do **not** prefix the version with a lowercase `v`.

## 1. Open the release issue.

Create a release issue using the template below. After creating it, pin it for easy access.

<details>
<summary><h3>Release Template</h3></summary>

#### Title:
```markdown
Release `YY.MM.MICRO`
```

#### Body:
```markdown
### Summary

Placeholder for `conda-libmamba-solver YY.MM.MICRO` release.

| Pilot | <pilot> |
|---|---|
| Co-pilot | <copilot> |

### Tasks

[milestone]: https://github.com/conda/conda-libmamba-solver/milestone/<milestone>
[process]: https://github.com/conda/conda-libmamba-solver/blob/main/RELEASE.md
[releases]: https://github.com/conda/conda-libmamba-solver/releases
[main]: https://github.com/AnacondaRecipes/conda-libmamba-solver-feedstock
[conda-forge]: https://github.com/conda-forge/conda-libmamba-solver-feedstock
[zulip]: https://conda.zulipchat.com/#narrow/channel/480811-releases

- [ ] [Complete outstanding PRs][milestone]
- [ ] Create release PR (see [release process][process])
- [ ] Create Zulip thread on [#releases][zulip]
    - [ ] Announce `YY.MM.MICRO` in-progress
- [ ] [Publish release][releases]
- [ ] Bump/update feedstocks
    - [ ] [Anaconda, Inc.'s feedstock][main]
    - [ ] [conda-forge feedstock][conda-forge]
- [ ] Hand off to packaging team(s)
- [ ] Announce release
    - [ ] Post on Zulip thread
```

</details>

## 2. Run rever.

<details>
<summary><h2>Rever steps</h2></summary>

Install [`rever`][rever docs] using whatever your project defines (e.g., a conda environment or a pixi `release` environment). For example, [conda-pypi](https://github.com/conda/conda-pypi) uses `pixi run -e release rever ...`; other projects may use `conda create -n rever conda-forge::rever` and run `rever` directly.

1. Clone and `cd` into the repository if you haven't done so already:

    ```bash
    $ git clone git@github.com:/conda-libmamba-solver.git
    $ cd conda-libmamba-solver
    ```

2. Fetch the latest changes and create a versioned branch off `main` for the release PR:

    ```bash
    $ git fetch upstream
    $ git switch -c changelog-YY.MM.MICRO --no-track upstream/main
    ```

3. Run `rever --activities authors --force YY.MM.MICRO`:

    > **Note:**
    > Include `--force` when re-running any rever command for the same version; without it, rever skips already-completed activities.

    ```bash
    $ rever --activities authors --force YY.MM.MICRO
    ```

    - If rever reports unknown authors, add or update entries in `.authors.yml` (new contributors get a new entry; existing contributors using a new name/email get an `aliases`/`alternate_emails` addition).

    - Verify the result with:

        ```bash
        $ git shortlog -se
        ```

      Compare this list against `AUTHORS.md` and repeat until they match.

4. Review news snippets in `news/` (use Markdown, **not** reStructuredText). Add snippets for any undocumented changes using the `news/TEMPLATE` as a guide, naming files `<PR #>-<short-slug>.md`.

    - You can utilize [GitHub's compare view][compare] to review what changes are to be included in this release.

    - Commit when satisfied:

        ```bash
        $ git add news/
        $ git commit -m "Update news"
        ```

5. Ensure the `[//]: # (current developments)` marker is present at the top of `CHANGELOG.md`, then run `rever --activities changelog --force YY.MM.MICRO`:

    ```bash
    $ rever --activities changelog --force YY.MM.MICRO
    ```

    - If this succeeds, undo the commit so both activities can be run together in the next step:

        ```bash
        $ git reset --hard HEAD~1
        ```

6. Run both activities together so the contributor list is embedded in the changelog entry:

    ```bash
    $ rever --force YY.MM.MICRO
    ```

7. Use [GitHub's auto-generated release notes][new release] to identify first-time contributors and add `made their first contribution in <URL>` next to their entry in the Contributors section of `CHANGELOG.md`. See [GitHub docs][release docs] for how to auto-generate the release notes. Commit:

    ```bash
    $ git add CHANGELOG.md
    $ git commit -m "Add first-time contributions"
    ```

8. Push the versioned branch:

    ```bash
    $ git push -u upstream
    ```

9. Open the Release PR targeting `main`:

    ```markdown
    ## Description

    ✂️ snip snip ✂️ the making of a new release.

    Xref #<RELEASE ISSUE>
    ```

10. [Create][new release] the release and **save as draft**:

    | Field | Value |
    |---|---|
    | Choose a tag | `YY.MM.MICRO` |
    | Target | `main` |
    | Body | copy/paste from `CHANGELOG.md` |

    > **Note:** Only publish the release after the release PR is merged.

</details>

## 3. Wait for review and approval of the release PR.

## 4. Merge the release PR and publish the release.

Go to the [releases page][new release], add the release notes from `CHANGELOG.md` to the draft, and publish.

## 5. Bump [Anaconda Recipes][Anaconda Recipes] and [conda-forge][conda-forge] feedstocks to use `YY.MM.MICRO`.

Open a PR to bump the Anaconda Recipes feedstock.

For conda-forge, the `regro-cf-autotick-bot` will usually open a PR automatically. Review and merge it (or push fixes to the autotick branch if needed).

> [!NOTE]
> Conda-forge's PRs will be auto-created via the `regro-cf-autotick-bot`. Follow the instructions below if any changes need to be made to the recipe that were not automatically added (these instructions are only necessary for anyone who is _not_ a conda-forge feedstock maintainer, since maintainers can push changes directly to the autotick branch):
> - Create a new branch based off of autotick's branch (autotick's branches usually use the `regro-cf-autotick-bot:XX.YY.[$patch_number]_[short hash]` syntax)
> - Add any changes via commits to that new branch
> - Open a new PR and push it against the `main` branch
>
> Make sure to include a comment on the original `autotick-bot` PR that a new pull request has been created, in order to avoid duplicating work!  `regro-cf-autotick-bot` will close the auto-created PR once the new PR is merged.
>
> For more information about this process, please read the ["Pushing to regro-cf-autotick-bot branch" section of the conda-forge documentation](https://conda-forge.org/docs/maintainer/updating_pkgs.html#pushing-to-regro-cf-autotick-bot-branch).

## 6. Hand off to Anaconda's packaging team.

> [!NOTE]
> This step should NOT be done past Thursday morning EST; please start the process on a Monday, Tuesday, or Wednesday instead in order to avoid any potential debugging sessions over evenings or weekends.

<details>
<summary>Internal process</summary>

1. Open packaging request in #package_requests Slack channel, include links to the Release PR and feedstock PRs.

2. Message packaging team/PM to let them know that a release has occurred and that you are the release manager.

</details>

## 7. Announce the release.

Post the release announcement on the Zulip thread in [#releases][zulip].
