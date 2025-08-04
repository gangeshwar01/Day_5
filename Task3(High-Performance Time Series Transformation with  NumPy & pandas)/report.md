# High-Performance Time Series Transformation Report

## Summary
This project compares various implementations (pandas, NumPy, and Numba) for efficient time-series transformations.

### Tasks Covered
- Rolling window stats (mean, variance)
- EWMA & covariance
- FFT-based spectral analysis and filtering

---

## Performance Benchmarks

| Rows      | Pandas Time (s) | NumPy Time (s) | Numba Time (s) |
|-----------|------------------|----------------|----------------|
| 10,000    | x.xxx            | x.xxx          | x.xxx          |
| 100,000   | x.xxx            | x.xxx          | x.xxx          |
| 1,000,000 | x.xxx            | x.xxx          | x.xxx          |

> (Actual times filled by `benchmark_results.csv`)

---

## Analysis
- **pandas**: Most user-friendly, optimized for many real-world use cases.
- **NumPy**: Efficient with stride tricks, but less flexible for variable-length rolling.
- **Numba**: Fastest for large data. Requires compilation overhead but worth it for big datasets.

---

## Recommendations
- Use `pandas.rolling()` for datasets < 100k rows.
- Use Numba or stride tricks for datasets ≥ 1M rows.
- Apply FFT only on evenly sampled signals.

---

## Auto-Selector Strategy (in practice)
```python
def auto_rolling_selector(data, window):
    if len(data) < 100_000:
        return rolling_stats_pandas(pd.DataFrame(data), window)
    else:
        return rolling_mean_numba(data, window)
