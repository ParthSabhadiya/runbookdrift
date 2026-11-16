# RunbookDrift: detecting stale runbook steps

Scripts behind the paper *RunbookGraph: Detecting Stale Operational Knowledge Before AI Agents Act on It*.

They build the RunbookDrift benchmark from the public version history of GitLab's production runbooks
(https://gitlab.com/gitlab-com/runbooks) and evaluate stale-step detectors with a temporal hold-out.

## Requirements

- git, Python 3.10+
- `pip install pandas numpy scikit-learn`

## Reproduce

```bash
git clone https://gitlab.com/gitlab-com/runbooks.git
python3 mine.py runbooks 2023-11-01 w2023.csv   # train window: labels observed by 2024-11-01
python3 mine.py runbooks 2024-11-01 w2024.csv   # test window:  labels observed by 2025-11-01
python3 evaluate.py maintained                  # writes results_maintained.json, scored_maintained.csv
python3 evaluate.py all                         # writes results_all.json
```

Snapshots are pinned to 00:00 UTC on the given date, so repeated runs on the same repository history produce
identical output. One window takes about one minute on a two-core machine.

## What the scripts do

- `mine.py`: extracts actionable steps (shell/SQL/Ruby/Python/PromQL code lines, command lines, and prose lines
  with inline code or links) from `docs/**/*.md` at snapshot T0, labels each step as `valid`, `drifted` or
  `reworded` by following the document to T0 + 12 months, and computes graph and history signals using only
  information available at T0.
- `evaluate.py`: trains on the 2023 window, tests on the 2024 window, and reports AUROC (with a
  document-clustered bootstrap CI), average precision, precision in the top 5% and recall in the top 10%.

## Label definition

- `valid`: the identical line still exists in the document 12 months later.
- `drifted`: the line changed or disappeared and at least one of its references (URL, host, path, command
  token) no longer appears in the document.
- `reworded`: the line changed but all references remain (excluded from evaluation).
- Documents deleted within the year are excluded; documents in which fewer than half of the steps survive are
  marked `rewritten` and excluded from the "maintained" scope.

## Use on your own runbooks

Point `mine.py` at any Git repository whose runbooks live under `docs/` as Markdown files (or edit
`md_files()` for a different layout). Labels come from your own history, so you can recalibrate yearly.
