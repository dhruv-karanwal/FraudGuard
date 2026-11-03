# FraudGuard: Financial Fraud Detection Using Deep Learning

Hybrid deep learning system that detects fraudulent transactions on the PaySim dataset,
with a Streamlit dashboard for live prediction and model performance.

## Status
Work in progress. Course project (Deep Learning, VIT Pune).

## Planned pipeline
- Data: PaySim (Kaggle `ealaxi/paysim1`), chronological split by `step`
- Model: DNN branch (transaction features) + GRU/LSTM branch (per-user history) with attention fusion
- Imbalance: class weights, threshold tuned on validation
- Evaluation: precision, recall, F1, ROC-AUC, PR-AUC, compared with LogReg, RF, XGBoost
- App: Streamlit dashboard (Dashboard, Predict Transaction, Model Performance)

## Setup
```bash
python -m venv .venv
pip install -r requirements.txt
```
Place the PaySim CSV in `data/raw/` (not tracked by git).
