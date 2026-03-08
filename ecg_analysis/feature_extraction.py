"""Feature extraction from preprocessed ECG signals.

Extracts time-domain (heart rate variability, R-peak statistics) and
frequency-domain (power spectral density band ratios) features that serve
as inputs to the ML classifier.
"""

import numpy as np
from scipy.signal import welch
from ecg_analysis.preprocessing import detect_r_peaks


def extract_time_domain_features(signal, r_peaks, fs=360):
    """Extract time-domain features from an ECG signal.

    Features include mean/std/min/max of RR intervals, heart rate
    statistics, signal amplitude statistics, and RMSSD (root mean square
    of successive RR differences).

    Parameters
    ----------
    signal : array-like
        Preprocessed ECG signal.
    r_peaks : array-like
        Indices of detected R-peaks.
    fs : int
        Sampling frequency in Hz.

    Returns
    -------
    dict
        Dictionary of time-domain feature names and values.
    """
    signal = np.asarray(signal, dtype=float)
    r_peaks = np.asarray(r_peaks)
    features = {}

    if len(r_peaks) > 1:
        rr_intervals = np.diff(r_peaks) / fs  # seconds
        heart_rates = 60.0 / rr_intervals

        features["mean_rr"] = np.mean(rr_intervals)
        features["std_rr"] = np.std(rr_intervals)
        features["min_rr"] = np.min(rr_intervals)
        features["max_rr"] = np.max(rr_intervals)
        features["mean_hr"] = np.mean(heart_rates)
        features["std_hr"] = np.std(heart_rates)
        features["min_hr"] = np.min(heart_rates)
        features["max_hr"] = np.max(heart_rates)

        rr_diffs = np.diff(rr_intervals)
        features["rmssd"] = np.sqrt(np.mean(rr_diffs ** 2))
    else:
        for key in ["mean_rr", "std_rr", "min_rr", "max_rr",
                     "mean_hr", "std_hr", "min_hr", "max_hr", "rmssd"]:
            features[key] = 0.0

    features["signal_mean"] = np.mean(signal)
    features["signal_std"] = np.std(signal)
    features["signal_min"] = np.min(signal)
    features["signal_max"] = np.max(signal)
    features["num_r_peaks"] = len(r_peaks)

    return features


def extract_frequency_domain_features(signal, fs=360):
    """Extract frequency-domain features via Welch's power spectral density.

    Computes total power and the power in the very-low-frequency (VLF),
    low-frequency (LF), and high-frequency (HF) bands, as well as the
    LF/HF ratio commonly used in HRV analysis.

    Parameters
    ----------
    signal : array-like
        Preprocessed ECG signal.
    fs : int
        Sampling frequency in Hz.

    Returns
    -------
    dict
        Dictionary of frequency-domain feature names and values.
    """
    signal = np.asarray(signal, dtype=float)
    nperseg = min(256, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    total_power = np.trapezoid(psd, freqs)

    vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.4)

    vlf_power = np.trapezoid(psd[vlf_mask], freqs[vlf_mask]) if vlf_mask.any() else 0.0
    lf_power = np.trapezoid(psd[lf_mask], freqs[lf_mask]) if lf_mask.any() else 0.0
    hf_power = np.trapezoid(psd[hf_mask], freqs[hf_mask]) if hf_mask.any() else 0.0

    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0.0

    return {
        "total_power": total_power,
        "vlf_power": vlf_power,
        "lf_power": lf_power,
        "hf_power": hf_power,
        "lf_hf_ratio": lf_hf_ratio,
    }


def extract_features(signal, fs=360, r_peaks=None):
    """Extract all features from an ECG signal.

    Combines time-domain and frequency-domain features into a single
    dictionary.

    Parameters
    ----------
    signal : array-like
        Preprocessed ECG signal.
    fs : int
        Sampling frequency in Hz.
    r_peaks : array-like or None
        Indices of detected R-peaks.  If ``None``, R-peaks are detected
        automatically.

    Returns
    -------
    dict
        Combined feature dictionary.
    """
    signal = np.asarray(signal, dtype=float)
    if r_peaks is None:
        r_peaks = detect_r_peaks(signal, fs=fs)

    time_features = extract_time_domain_features(signal, r_peaks, fs=fs)
    freq_features = extract_frequency_domain_features(signal, fs=fs)

    return {**time_features, **freq_features}
