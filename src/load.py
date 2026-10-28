import pandas as pd

from .config import RAW

KEYS = ["stock_id", "date_id", "seconds_in_bucket"]


def load_raw(path=RAW):
    return pd.read_csv(path)
