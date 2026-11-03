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

## Setup and commands
- Venv: `python -m venv .venv`, then `.venv/Scripts/pip install -r requirements.txt` (Windows).
- Raw data: `data/raw/paysim.csv`. First load caches `data/processed/paysim_full.parquet`.
- Tests: `.venv/Scripts/python -m pytest -q`
- Stage 1 EDA: `python -m src.eda` → `reports/eda_summary.json`, `reports/figures/{class_distribution,fraud_over_time,amount_distribution}.png`
- Stage 2 arrays: `python -m src.prepare --dev` (sample) or `python -m src.prepare` (full)
- Stage 3 baselines: `python -m src.baselines [--dev] [--only xgboost ...]` → `models/<tag>/*.joblib|*.keras`,
  `*_val_scores.npy`, `reports/<tag>_val_metrics.json` (validation only; full run ~4 min, RF dominates)
- Stage 4 hybrid: `python -m src.train [--dev] [--rnn gru|lstm]` → `models/<tag>/hybrid_<rnn>.keras` (+ history,
  val scores). Full run ~4 min per model on CPU (TF has no GPU on native Windows).
- Stage 5 eval: `python -m src.evaluate [--dev]` scores test ONCE (marker `models/<tag>/test_evaluated.json`
  blocks re-runs; `--force` only knowingly). `--figures-only` redraws figures from saved test scores.
  Outputs `reports/<tag>_test_metrics.json`, `reports/hybrid_model_summary.txt`, figures: model_comparison,
  roc_pr_curves, training_curves, confusion_matrix_*, architecture.
- Stage 6 app: `streamlit run app/streamlit_app.py` (pages: Dashboard, Predict Transaction, Model Performance).
  Uses `models/full` if present, else `models/dev`. Tests: `tests/test_app.py` (AppTest, skipped without artifacts).
  On Windows set `PYTHONIOENCODING=utf-8` when printing page titles (emoji).

## Code map
- `README.md`: results table and findings; numbers come from `reports/full_test_metrics.json`. Update both together.
- `src/config.py`: paths, `SEED=42`, split fractions (70/15/15 of scored rows), `SEQ_LEN`, dev-sample rate.
- `src/data.py`: `load_paysim` (adds `row_id` = file order), `dev_sample` (all fraud + every k-th legit row),
  `chronological_split` (sorts by `step,row_id`, boundaries on whole steps, computed on TRANSFER/CASH_OUT rows).
- `src/eda.py`: EDA stats and figures.
- `src/features.py`: `raw_features` (row-local: log amounts/balances, balance-error terms, flags, hour sin/cos,
  type one-hots); `TAB_FEATURES` (14, DNN branch) and `SEQ_FEATURES` (12, recurrent branch);
  `FeatureScaler` (StandardScaler on continuous columns only, fit on train).
- `src/sequences.py`: `history_index` (last `SEQ_LEN` rows of the same account, current row last, left-padded,
  grouped by (split, key) so windows never cross a split boundary) and `gather_sequences`
  (adds a `log1p(hours since)` channel, zero padding + mask).
- `src/prepare.py`: `python -m src.prepare [--dev]` → `data/processed/<dev|full>/{split}_{tab,seq,mask,y}.npy`,
  `{split}_meta.parquet`; scalers + `prep_meta.json` in `models/<dev|full>/`. Full run takes ~15 s.
  History windows use all transaction types; only TRANSFER/CASH_OUT rows are emitted.
- `src/metrics.py`: `evaluate` (all metrics + CM `[[TN, FP], [FN, TP]]`), `best_f1_threshold` (use on val only),
  `class_weights` (balanced).
- `src/models.py`: `build_dnn`, `build_hybrid` (DNN branch + GRU/LSTM with `MaskedAttentionPool`, fused by
  `FusionAttention`), `explainer` (prob + temporal + fusion weights). Custom layers are keras-serializable.
- `src/training.py`: `set_seed`, `fit_keras` (class weights, early stop on `val_pr_auc`, saves history json).
- `src/train.py`: trains the hybrid with class weights; writes val metrics into `reports/<tag>_val_metrics.json`.
- `src/inference.py`: `Predictor` (raw fields + optional history → prob, threshold, temporal/fusion
  attention); must match the batch pipeline exactly (`tests/test_inference.py`).
- `app/streamlit_app.py`: three-page dashboard; reads figures/metrics from `reports/` and test meta/scores.
- `src/evaluate.py`: thresholds from saved val scores, single test scoring, all report figures.
- `src/baselines.py`: LogReg, RandomForest, XGBoost (scale_pos_weight, early stop on val aucpr), DNN-only.

## Data facts (from EDA)
- 6,362,620 rows, 8,213 fraud (0.129%). Fraud only in TRANSFER (4,097) and CASH_OUT (4,116); models score only those types.
- `isFlaggedFraud` flags 16 rows, all fraud: label leak, never a feature.
- `nameOrig` almost never repeats: 9,298 of 6.35M accounts have >1 txn (max 3). Per-user history is very sparse.
- Split boundaries: train steps 1-323, val 324-377, test 378-743. Legit volume collapses after ~step 400 while
  fraud stays flat, so test fraud rate (~1%) is much higher than train (~0.19%).

## Model findings
- PaySim is near-separable from balance columns: 98% of fraud drains the origin exactly (amount = oldbalanceOrg),
  0.0% of legit. Tree models hit F1 = 1.0 on validation.
- Full test (`models/full`, evaluated once): RF F1 0.9979, XGB 0.9967, Hybrid-LSTM 0.9959, Hybrid-GRU 0.9949,
  DNN-only 0.9939, LogReg 0.9888. Hybrid beats DNN-only (fewer missed frauds) but not the tree models.
- The hybrid puts ~10% fusion weight on the history branch (3% on fraud rows): `nameOrig` history is almost empty.

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
