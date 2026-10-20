import hashlib, numpy as np
p = np.random.default_rng(20260927).choice(40_180_608, size=5_000_000, replace=False).astype(np.int64)
print(np.__version__, hashlib.sha256(p.tobytes()).hexdigest(), int(p.min()), int(p.max()), len(np.unique(p)))
