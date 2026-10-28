from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "train.csv"
PROCESSED = ROOT / "data" / "processed" / "clean.parquet"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

ALPHA = 0.05
SEED = 558
N_BOOT = 2000

BOUNDARY = 300
TEST_SECONDS = tuple(range(0, 541, 60))
T2_SECOND = 0
WINSOR = (0.001, 0.999)
CHEBYSHEV_K = 10

MATTERS_CRAMER_V = 0.10
MATTERS_DELTA_IC = 0.02
MATTERS_VAR_RATIO = (0.9, 1 / 0.9)
MATTERS_HIT_RATE = 0.52
