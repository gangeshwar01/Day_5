import numpy as np
import pandas as pd
from numba import njit

def rolling_stats_numpy(arr, window):
    shape = (arr.shape[0] - window + 1, window, arr.shape[1])
    strides = (arr.strides[0], arr.strides[0], arr.strides[1])
    windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    mean = windows.mean(axis=1)
    var = windows.var(axis=1)
    return mean, var

def rolling_stats_pandas(df, window):
    return df.rolling(window).mean(), df.rolling(window).var()

def ewma_numpy(data, alpha):
    ewma = np.zeros_like(data)
    ewma[0] = data[0]
    for t in range(1, len(data)):
        ewma[t] = alpha * data[t] + (1 - alpha) * ewma[t - 1]
    return ewma

def ewma_pandas(df, alpha):
    return df.ewm(alpha=alpha).mean()

def fft_spectral_analysis(data):
    fft_vals = np.fft.fft(data, axis=0)
    freqs = np.fft.fftfreq(data.shape[0])
    return freqs, np.abs(fft_vals)

def bandpass_filter(data, low_freq, high_freq, sample_rate):
    fft_vals = np.fft.fft(data, axis=0)
    freqs = np.fft.fftfreq(data.shape[0], d=1/sample_rate)
    mask = (np.abs(freqs) >= low_freq) & (np.abs(freqs) <= high_freq)
    fft_vals[~mask] = 0
    return np.real(np.fft.ifft(fft_vals, axis=0))

@njit
def rolling_mean_numba(arr, window):
    n, m = arr.shape
    result = np.empty((n - window + 1, m))
    for i in range(n - window + 1):
        for j in range(m):
            result[i, j] = np.mean(arr[i:i + window, j])
    return result
