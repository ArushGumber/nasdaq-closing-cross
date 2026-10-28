import numpy as np

PRIME = 2_147_483_647
EPS = 0.001
DELTA = 0.01
TOP = 10


class CountMin:
    def __init__(self, eps=EPS, delta=DELTA, seed=0):
        self.width = int(np.ceil(2 / eps))
        self.depth = int(np.ceil(np.log2(1 / delta)))
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, PRIME, self.depth)
        self.b = rng.integers(0, PRIME, self.depth)
        self.table = np.zeros((self.depth, self.width))

    def buckets(self, keys):
        keys = np.asarray(keys, dtype="int64")
        return ((self.a[:, None] * keys[None, :] + self.b[:, None]) % PRIME) % self.width

    def update(self, keys, weights):
        for row, cols in enumerate(self.buckets(keys)):
            np.add.at(self.table[row], cols, weights)

    def query(self, keys):
        cols = self.buckets(keys)
        return self.table[np.arange(self.depth)[:, None], cols].min(axis=0)


def heavy_stock_days(df, chunk=500_000):
    ordered = df.sort_values(["date_id", "seconds_in_bucket", "stock_id"])
    keys = ordered["date_id"].to_numpy().astype("int64") * 1000 + ordered["stock_id"].to_numpy()
    weights = ordered["imbalance_size"].to_numpy().astype("float64")
    sketch = CountMin()
    for start in range(0, len(keys), chunk):
        sketch.update(keys[start:start + chunk], weights[start:start + chunk])

    unique, inverse = np.unique(keys, return_inverse=True)
    exact = np.bincount(inverse, weights=weights)
    estimate = sketch.query(unique)
    error = estimate - exact
    total = weights.sum()
    bound = EPS * total
    top_exact = set(unique[np.argsort(exact)[-TOP:]])
    top_sketch = set(unique[np.argsort(estimate)[-TOP:]])
    return {
        "rows": int(len(keys)), "items": int(len(unique)), "width": sketch.width, "depth": sketch.depth,
        "counters": int(sketch.width * sketch.depth), "total_weight": float(total), "bound": float(bound),
        "max_error": float(error.max()), "min_error": float(error.min()),
        "share_within_bound": float((error <= bound).mean()), "top_overlap": len(top_exact & top_sketch),
        "median_relative_error_top": float(np.median(error[np.argsort(exact)[-100:]] / exact[np.argsort(exact)[-100:]])),
        "error_share_of_bound": np.round(np.sort(error / bound), 4).tolist(),
    }
