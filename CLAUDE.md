# CLAUDE.md

Project guide for AI-assisted work. Update this file whenever code or structure changes.

## Project
FraudGuard: hybrid DNN + GRU/LSTM (attention fusion) fraud detector on PaySim, plus a Streamlit app.

## Layout
- `src/`: data, features, models, training, evaluation code
- `app/`: Streamlit dashboard
- `notebooks/`: exploration only
- `tests/`: pytest tests
- `data/raw/`: PaySim CSV (git-ignored)
- `models/`, `reports/figures/`: generated outputs (git-ignored)

## Rules
- Chronological split by `step`; fit scalers/encoders on train only; tune threshold on validation; touch test once.
- Fixed random seed (42) everywhere.
- Report precision, recall, F1, ROC-AUC, PR-AUC. Accuracy alone is not meaningful here.

## Workflow: nested loops
- Outer loop: stage (data, features, baselines, hybrid model, app, report).
- Inner loop per stage: implement, run, check metrics/tests, fix, repeat until the stage passes.
- Commit after each passing stage, not before.

## Git conventions
- Conventional Commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`.
- Small, focused commits; imperative subject under 72 chars; body explains why.
- Commits are authored by the user only. No AI co-author or attribution lines.
  A `commit-msg` hook in `.git/hooks` enforces this locally.
