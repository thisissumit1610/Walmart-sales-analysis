"""Generate synthetic ECG-like signals for demonstration and testing.

Real ECG data (e.g. MIT-BIH Arrhythmia Database) should be used for
production models.  The synthetic signals produced here mimic basic ECG
morphology and are labelled with disease classes so that the full ML
pipeline can be exercised without external data dependencies.
"""

import numpy as np
import pandas as pd
from ecg_analysis.preprocessing import preprocess_ecg
from ecg_analysis.feature_extraction import extract_features


def generate_single_ecg(duration=10, fs=360, heart_rate=72, noise_level=0.05,
                         abnormality=None, random_state=None):
    """Generate a single synthetic ECG-like signal.

    Parameters
    ----------
    duration : float
        Signal duration in seconds.
    fs : int
        Sampling frequency in Hz.
    heart_rate : int
        Approximate heart rate in beats per minute.
    noise_level : float
        Standard deviation of additive Gaussian noise.
    abnormality : str or None
        Type of simulated abnormality.  One of ``None`` (normal),
        ``"arrhythmia"``, ``"mi"`` (myocardial infarction),
        ``"afib"`` (atrial fibrillation), or ``"bradycardia"``.
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Synthetic ECG signal.
    """
    rng = np.random.RandomState(random_state)
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    signal = np.zeros(n_samples)

    if abnormality == "bradycardia":
        heart_rate = rng.randint(40, 55)
    elif abnormality == "arrhythmia":
        heart_rate = heart_rate + rng.randint(-15, 15)

    beat_interval = 60.0 / heart_rate
    beat_times = np.arange(0, duration, beat_interval)

    if abnormality == "arrhythmia":
        beat_times = beat_times + rng.uniform(-0.1, 0.1, len(beat_times))
    elif abnormality == "afib":
        beat_times = beat_times + rng.uniform(-0.2, 0.2, len(beat_times))

    for bt in beat_times:
        # P wave
        p_center = bt - 0.15
        p_mask = np.abs(t - p_center) < 0.06
        if abnormality == "afib":
            signal[p_mask] += 0.05 * rng.uniform(0.5, 1.5)
        else:
            signal[p_mask] += 0.15 * np.exp(-((t[p_mask] - p_center) ** 2) / (2 * 0.02 ** 2))

        # QRS complex
        qrs_center = bt
        q_mask = np.abs(t - (qrs_center - 0.03)) < 0.015
        signal[q_mask] -= 0.1

        r_mask = np.abs(t - qrs_center) < 0.02
        r_amplitude = 1.0
        if abnormality == "mi":
            r_amplitude = rng.uniform(0.3, 0.6)
        signal[r_mask] += r_amplitude * np.exp(-((t[r_mask] - qrs_center) ** 2) / (2 * 0.008 ** 2))

        s_mask = np.abs(t - (qrs_center + 0.03)) < 0.015
        signal[s_mask] -= 0.15

        # T wave
        t_center = bt + 0.2
        t_mask = np.abs(t - t_center) < 0.08
        t_amp = 0.3
        if abnormality == "mi":
            t_amp = rng.uniform(-0.3, -0.1)  # inverted T wave
        signal[t_mask] += t_amp * np.exp(-((t[t_mask] - t_center) ** 2) / (2 * 0.04 ** 2))

    signal += rng.normal(0, noise_level, n_samples)
    return signal


ABNORMALITY_LABEL = {
    None: 0,
    "arrhythmia": 1,
    "mi": 2,
    "afib": 3,
    "bradycardia": 4,
}


def generate_dataset(n_samples_per_class=50, duration=10, fs=360, random_state=42):
    """Generate a labelled dataset of synthetic ECG signals.

    Parameters
    ----------
    n_samples_per_class : int
        Number of signals per disease class.
    duration : float
        Duration of each signal in seconds.
    fs : int
        Sampling frequency in Hz.
    random_state : int
        Base random seed.

    Returns
    -------
    signals : list of numpy.ndarray
        Raw ECG signals.
    labels : numpy.ndarray
        Integer disease labels.
    """
    rng = np.random.RandomState(random_state)
    signals = []
    labels = []

    for abnormality, label in ABNORMALITY_LABEL.items():
        for i in range(n_samples_per_class):
            hr = rng.randint(60, 100) if abnormality not in ("bradycardia",) else 72
            noise = rng.uniform(0.02, 0.08)
            sig = generate_single_ecg(
                duration=duration,
                fs=fs,
                heart_rate=hr,
                noise_level=noise,
                abnormality=abnormality,
                random_state=random_state + label * 1000 + i,
            )
            signals.append(sig)
            labels.append(label)

    return signals, np.array(labels)


def generate_feature_dataset(n_samples_per_class=50, duration=10, fs=360, random_state=42):
    """Generate a feature matrix and labels from synthetic ECG signals.

    Convenience wrapper that calls :func:`generate_dataset`, preprocesses
    each signal, and extracts features.

    Parameters
    ----------
    n_samples_per_class : int
        Number of signals per disease class.
    duration : float
        Duration of each signal in seconds.
    fs : int
        Sampling frequency in Hz.
    random_state : int
        Base random seed.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix.
    y : numpy.ndarray
        Integer disease labels.
    """
    signals, labels = generate_dataset(
        n_samples_per_class=n_samples_per_class,
        duration=duration,
        fs=fs,
        random_state=random_state,
    )

    feature_dicts = []
    for sig in signals:
        preprocessed = preprocess_ecg(sig, fs=fs)
        features = extract_features(
            preprocessed["normalized"], fs=fs, r_peaks=preprocessed["r_peaks"]
        )
        feature_dicts.append(features)

    return pd.DataFrame(feature_dicts), labels
