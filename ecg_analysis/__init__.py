"""ECG Signal Analysis and ML-based Disease Prediction.

This package provides tools for processing ECG (Electrocardiogram) signals
and using machine learning to predict cardiac conditions.

Modules:
    preprocessing: Signal filtering, normalization, and R-peak detection.
    feature_extraction: Extract time-domain and frequency-domain features.
    model: Train, evaluate, and use ML models for disease prediction.
    generate_synthetic_data: Create synthetic ECG data for demonstration.
"""

from ecg_analysis.preprocessing import preprocess_ecg, bandpass_filter, normalize_signal, detect_r_peaks
from ecg_analysis.feature_extraction import extract_features, extract_time_domain_features, extract_frequency_domain_features
from ecg_analysis.model import ECGClassifier
