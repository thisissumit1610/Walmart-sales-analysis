"""ECG signal preprocessing utilities.

Provides bandpass filtering, normalization, and R-peak detection
to prepare raw ECG signals for feature extraction.
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def bandpass_filter(signal, lowcut=0.5, highcut=45.0, fs=360, order=4):
    """Apply a Butterworth bandpass filter to an ECG signal.

    Parameters
    ----------
    signal : array-like
        Raw ECG signal.
    lowcut : float
        Lower cutoff frequency in Hz.
    highcut : float
        Upper cutoff frequency in Hz.
    fs : int
        Sampling frequency in Hz.
    order : int
        Filter order.

    Returns
    -------
    numpy.ndarray
        Filtered ECG signal.
    """
    signal = np.asarray(signal, dtype=float)
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def normalize_signal(signal):
    """Normalize an ECG signal to zero mean and unit variance.

    Parameters
    ----------
    signal : array-like
        ECG signal to normalize.

    Returns
    -------
    numpy.ndarray
        Normalized signal.
    """
    signal = np.asarray(signal, dtype=float)
    std = np.std(signal)
    if std == 0:
        return signal - np.mean(signal)
    return (signal - np.mean(signal)) / std


def detect_r_peaks(signal, fs=360, height_factor=0.6, min_distance_ms=200):
    """Detect R-peaks in an ECG signal.

    Uses scipy's ``find_peaks`` with adaptive height and minimum distance
    thresholds derived from the signal and sampling rate.

    Parameters
    ----------
    signal : array-like
        Preprocessed ECG signal.
    fs : int
        Sampling frequency in Hz.
    height_factor : float
        Fraction of the maximum signal amplitude used as the minimum peak
        height threshold.
    min_distance_ms : int
        Minimum distance between consecutive R-peaks in milliseconds.

    Returns
    -------
    numpy.ndarray
        Indices of detected R-peaks.
    """
    signal = np.asarray(signal, dtype=float)
    min_distance_samples = int(min_distance_ms * fs / 1000)
    height_threshold = height_factor * np.max(signal)
    peaks, _ = find_peaks(signal, height=height_threshold, distance=min_distance_samples)
    return peaks


def preprocess_ecg(signal, fs=360, lowcut=0.5, highcut=45.0):
    """Full preprocessing pipeline: filter, normalize, and detect R-peaks.

    Parameters
    ----------
    signal : array-like
        Raw ECG signal.
    fs : int
        Sampling frequency in Hz.
    lowcut : float
        Lower cutoff frequency in Hz.
    highcut : float
        Upper cutoff frequency in Hz.

    Returns
    -------
    dict
        Dictionary with keys ``filtered``, ``normalized``, and ``r_peaks``.
    """
    filtered = bandpass_filter(signal, lowcut=lowcut, highcut=highcut, fs=fs)
    normalized = normalize_signal(filtered)
    r_peaks = detect_r_peaks(normalized, fs=fs)
    return {
        "filtered": filtered,
        "normalized": normalized,
        "r_peaks": r_peaks,
    }
