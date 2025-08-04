# High-Performance Time Series Transformation with NumPy & pandas

## 📌 Project Overview

This project focuses on building **efficient, scalable, and high-performance routines** for transforming large multivariate time-series datasets (1M+ rows). Using both `NumPy` and `pandas`, along with `Numba` and `stride tricks`, the goal is to optimize:

* Rolling window statistics
* Exponentially Weighted Moving Averages (EWMA)
* Spectral analysis via Fast Fourier Transform (FFT)

It includes **runtime and memory benchmarks**, and dynamically selects the best-performing method depending on the dataset characteristics.

---

## 🚀 Features

### ✅ Core Functionalities

1. **Rolling Statistics**

   * Rolling mean and variance
   * Arbitrary window sizes
2. **EWMA and Covariance**

   * Efficiently computed using both pandas and vectorized NumPy
3. **FFT-Based Analysis**

   * Spectral density estimation
   * Band-pass filtering

### ⚡ Performance Optimizations

* Pure NumPy vs. pandas comparison
* Acceleration using:

  * **NumPy stride tricks**
  * **Numba JIT compilation**
* Smart method selection based on dataset size and memory constraints

---

## 🧠 File Structure

```
📁 project-root/
│
├── timeseries_utils.py       # Main transformation functions (NumPy/pandas/accelerated)
├── benchmark.py              # Script to run performance benchmarks
├── benchmark_results.csv     # Logged runtime and memory usage metrics
├── report.md                 # Summary report of performance trade-offs
└── README.md                 # Project documentation (this file)
```

---

## 🧪 Benchmarking & Evaluation

* Compares:

  * `pandas` built-ins (`rolling`, `ewm`)
  * Manual `NumPy` implementations
  * Accelerated versions (`stride tricks`, `Numba`)
* Outputs performance data as CSV + visual charts
* Includes:

  * Time vs. Memory usage trade-offs
  * Heatmaps and bar plots
  * Recommendations per use case

---

## 🛠️ Requirements

* Python ≥ 3.8
* NumPy
* pandas
* Numba
* matplotlib / seaborn (for plotting)

```bash
pip install numpy pandas numba matplotlib seaborn
```

---

## 📈 Usage

### 1. Compute Rolling Statistics

```python
from timeseries_utils import rolling_mean_numpy, rolling_mean_pandas

mean_arr = rolling_mean_numpy(data, window=100)
```

### 2. Run Benchmarks

```bash
python benchmark.py
```

### 3. Auto-select Fastest Method

```python
from timeseries_utils import auto_select_rolling

best_result = auto_select_rolling(data, window=200)
```

---

## 📊 Report

The report includes:

* Benchmarks across multiple dataset sizes (1M – 100M rows)
* Graphs comparing execution time & memory usage
* Recommendations for choosing between NumPy, pandas, or accelerated variants
* Code profiling summaries (e.g., using `memory_profiler`, `timeit`, or `perf_counter`)

---

## ✅ Deliverables

* [x] `timeseries_utils.py` with all implementations
* [x] `benchmark.py` script
* [x] `benchmark_results.csv`
* [x] Report in `.md`
* [x] This `README.md`

---

## 📚 References

* [NumPy Documentation](https://numpy.org/doc/)
* [pandas Rolling and EWMA](https://pandas.pydata.org/pandas-docs/stable/user_guide/window.html)
* [Numba Guide](https://numba.pydata.org/)

---

