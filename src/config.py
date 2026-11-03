"""Project-wide paths and constants."""
from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "paysim.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"

# Fraud only occurs in these types; the models score only these.
FRAUD_TYPES = ("TRANSFER", "CASH_OUT")
ALL_TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")

# Chronological split: fractions of rows (ordered by step) for train / val;
# the remainder is test. Boundaries fall on whole steps.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# Length of the per-user history window fed to the GRU branch.
SEQ_LEN = 5

# Development sample: keep every k-th legit row (in time order) plus all fraud.
DEV_LEGIT_EVERY = 10

TARGET = "isFraud"
# isFlaggedFraud is derived from the label by the simulator; never a feature.
LEAKY_COLS = ("isFlaggedFraud",)


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
