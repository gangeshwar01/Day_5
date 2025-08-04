import time
import numpy as np
import pandas as pd
from timeseries_utils import rolling_stats_numpy, rolling_stats_pandas, rolling_mean_numba

sizes = [10_000, 100_000, 1_000_000]
window = 50
cols = 3
results = []

for size in sizes:
    data = np.random.randn(size, cols)
    df = pd.DataFrame(data)

    start = time.time()
    rolling_stats_pandas(df, window)
    t_pandas = time.time() - start

    start = time.time()
    rolling_stats_numpy(data, window)
    t_numpy = time.time() - start

    start = time.time()
    rolling_mean_numba(data, window)
    t_numba = time.time() - start

    results.append({
        "Rows": size,
        "Pandas Time (s)": round(t_pandas, 4),
        "NumPy Time (s)": round(t_numpy, 4),
        "Numba Time (s)": round(t_numba, 4),
    })

pd.DataFrame(results).to_csv("benchmark_results.csv", index=False)
print(pd.DataFrame(results))
