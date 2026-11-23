"""Central configuration. All paths are relative to the project root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
OUTPUT_DIR = ROOT / "outputs"

DATASET = "FD001"
SEQ_LEN = 30          # cycles per input window (configurable via train.py --seq-len)
RUL_CLIP = 125        # piece-wise linear RUL cap (configurable via train.py --rul-clip)
VAL_FRACTION = 0.2    # fraction of training ENGINES held out for validation
SEED = 42

# Health states are data-driven: see src/health_state.py and data/processed/health_state_params.json.

LSTM_CHECKPOINT = CHECKPOINT_DIR / "lstm_best.pt"
PREPROCESSOR_PATH = PROCESSED_DIR / "preprocessor.joblib"
HISTORY_PATH = OUTPUT_DIR / "training_history.json"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
TEST_PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"

# Seed of the local generator that draws the MC-dropout masks (uncertainty only; point predictions
# never use dropout). Fixed so identical inputs give identical uncertainty estimates.
MC_DROPOUT_SEED = 0
MC_DROPOUT_SAMPLES = 30
