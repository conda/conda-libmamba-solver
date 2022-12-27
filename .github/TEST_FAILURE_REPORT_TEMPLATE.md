---
title: "{{ env.TITLE }}"
labels: [bug]
---

The {{ workflow }} workflow failed on {{ date | date("YYYY-MM-DD HH:mm") }} UTC

Full run: https://github.com/conda/conda-libmamba-solver/actions/runs/{{ env.RUN_ID }}

(This post will be updated if another test fails, as long as this issue remains open.)