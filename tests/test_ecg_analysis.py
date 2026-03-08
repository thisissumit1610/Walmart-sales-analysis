"""Unit tests for the ecg_analysis package."""

import numpy as np
import pandas as pd
import pytest
import os
import tempfile

from ecg_analysis.preprocessing import bandpass_filter, normalize_signal, detect_r_peaks, preprocess_ecg
from ecg_analysis.feature_extraction import extract_time_domain_features, extract_frequency_domain_features, extract_features
from ecg_analysis.model import ECGClassifier, DISEASE_LABELS
from ecg_analysis.generate_synthetic_data import generate_single_ecg, generate_dataset, generate_feature_dataset


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestBandpassFilter:
    def test_output_length(self):
        signal = np.random.randn(3600)
        filtered = bandpass_filter(signal, fs=360)
        assert len(filtered) == len(signal)

    def test_removes_dc_offset(self):
        signal = np.ones(3600) * 5 + np.random.randn(3600) * 0.1
        filtered = bandpass_filter(signal, fs=360)
        assert abs(np.mean(filtered)) < abs(np.mean(signal))


class TestNormalizeSignal:
    def test_zero_mean(self):
        signal = np.random.randn(1000) * 3 + 5
        normalized = normalize_signal(signal)
        assert abs(np.mean(normalized)) < 1e-10

    def test_unit_variance(self):
        signal = np.random.randn(1000) * 3 + 5
        normalized = normalize_signal(signal)
        assert abs(np.std(normalized) - 1.0) < 1e-10

    def test_constant_signal(self):
        signal = np.ones(100) * 42
        normalized = normalize_signal(signal)
        assert np.allclose(normalized, 0.0)


class TestDetectRPeaks:
    def test_finds_peaks_in_synthetic_signal(self):
        ecg = generate_single_ecg(duration=5, fs=360, heart_rate=72, random_state=0)
        normalized = normalize_signal(ecg)
        peaks = detect_r_peaks(normalized, fs=360)
        expected_beats = int(5 * 72 / 60)
        assert len(peaks) >= expected_beats - 2
        assert len(peaks) <= expected_beats + 2


class TestPreprocessECG:
    def test_returns_expected_keys(self):
        ecg = generate_single_ecg(duration=3, fs=360, random_state=0)
        result = preprocess_ecg(ecg, fs=360)
        assert "filtered" in result
        assert "normalized" in result
        assert "r_peaks" in result

    def test_output_shapes(self):
        ecg = generate_single_ecg(duration=3, fs=360, random_state=0)
        result = preprocess_ecg(ecg, fs=360)
        assert len(result["filtered"]) == len(ecg)
        assert len(result["normalized"]) == len(ecg)


# ---------------------------------------------------------------------------
# Feature extraction tests
# ---------------------------------------------------------------------------

class TestTimeDomainFeatures:
    def test_returns_all_keys(self):
        ecg = generate_single_ecg(duration=5, fs=360, random_state=1)
        result = preprocess_ecg(ecg, fs=360)
        features = extract_time_domain_features(result["normalized"], result["r_peaks"], fs=360)
        expected_keys = {"mean_rr", "std_rr", "min_rr", "max_rr",
                         "mean_hr", "std_hr", "min_hr", "max_hr",
                         "rmssd", "signal_mean", "signal_std", "signal_min",
                         "signal_max", "num_r_peaks"}
        assert set(features.keys()) == expected_keys

    def test_heart_rate_range(self):
        ecg = generate_single_ecg(duration=10, fs=360, heart_rate=72, random_state=2)
        result = preprocess_ecg(ecg, fs=360)
        features = extract_time_domain_features(result["normalized"], result["r_peaks"], fs=360)
        assert 30 < features["mean_hr"] < 200


class TestFrequencyDomainFeatures:
    def test_returns_all_keys(self):
        ecg = generate_single_ecg(duration=5, fs=360, random_state=3)
        result = preprocess_ecg(ecg, fs=360)
        features = extract_frequency_domain_features(result["normalized"], fs=360)
        expected_keys = {"total_power", "vlf_power", "lf_power", "hf_power", "lf_hf_ratio"}
        assert set(features.keys()) == expected_keys

    def test_total_power_positive(self):
        ecg = generate_single_ecg(duration=5, fs=360, random_state=4)
        result = preprocess_ecg(ecg, fs=360)
        features = extract_frequency_domain_features(result["normalized"], fs=360)
        assert features["total_power"] > 0


class TestExtractFeatures:
    def test_combined_feature_count(self):
        ecg = generate_single_ecg(duration=5, fs=360, random_state=5)
        result = preprocess_ecg(ecg, fs=360)
        features = extract_features(result["normalized"], fs=360, r_peaks=result["r_peaks"])
        assert len(features) == 19  # 14 time-domain + 5 frequency-domain


# ---------------------------------------------------------------------------
# Synthetic data generation tests
# ---------------------------------------------------------------------------

class TestGenerateSingleECG:
    def test_output_length(self):
        sig = generate_single_ecg(duration=5, fs=360)
        assert len(sig) == 5 * 360

    def test_different_abnormalities(self):
        for abn in [None, "arrhythmia", "mi", "afib", "bradycardia"]:
            sig = generate_single_ecg(duration=3, fs=360, abnormality=abn, random_state=10)
            assert len(sig) == 3 * 360


class TestGenerateDataset:
    def test_dataset_size(self):
        signals, labels = generate_dataset(n_samples_per_class=5, duration=3, random_state=0)
        assert len(signals) == 5 * 5  # 5 classes * 5 samples
        assert len(labels) == 25

    def test_label_distribution(self):
        _, labels = generate_dataset(n_samples_per_class=10, duration=3, random_state=0)
        for label_id in range(5):
            assert np.sum(labels == label_id) == 10


class TestGenerateFeatureDataset:
    def test_returns_dataframe(self):
        X, y = generate_feature_dataset(n_samples_per_class=3, duration=3, random_state=0)
        assert isinstance(X, pd.DataFrame)
        assert len(X) == 15
        assert len(y) == 15


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestECGClassifier:
    @pytest.fixture
    def small_dataset(self):
        X, y = generate_feature_dataset(n_samples_per_class=10, duration=5, random_state=42)
        return X, y

    def test_invalid_model_type(self):
        with pytest.raises(ValueError, match="Unsupported model type"):
            ECGClassifier(model_type="invalid")

    def test_train_and_predict(self, small_dataset):
        X, y = small_dataset
        clf = ECGClassifier(model_type="random_forest", n_estimators=10, random_state=42)
        clf.train(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(y)

    def test_evaluate(self, small_dataset):
        X, y = small_dataset
        clf = ECGClassifier(model_type="random_forest", n_estimators=10, random_state=42)
        clf.train(X, y)
        result = clf.evaluate(X, y)
        assert "accuracy" in result
        assert "classification_report" in result
        assert "confusion_matrix" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_predict_disease_labels(self, small_dataset):
        X, y = small_dataset
        clf = ECGClassifier(model_type="random_forest", n_estimators=10, random_state=42)
        clf.train(X, y)
        disease_names = clf.predict_disease(X)
        assert all(isinstance(name, str) for name in disease_names)

    def test_save_and_load(self, small_dataset):
        X, y = small_dataset
        clf = ECGClassifier(model_type="random_forest", n_estimators=10, random_state=42)
        clf.train(X, y)
        preds_before = clf.predict(X)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            filepath = f.name
        try:
            clf.save_model(filepath)
            loaded = ECGClassifier.load_model(filepath, model_type="random_forest")
            preds_after = loaded.predict(X)
            np.testing.assert_array_equal(preds_before, preds_after)
        finally:
            os.unlink(filepath)

    def test_cross_validate(self, small_dataset):
        X, y = small_dataset
        clf = ECGClassifier(model_type="random_forest", n_estimators=10, random_state=42)
        result = clf.cross_validate(X, y, cv=3)
        assert "mean_accuracy" in result
        assert "std_accuracy" in result

    def test_gradient_boosting(self, small_dataset):
        X, y = small_dataset
        clf = ECGClassifier(model_type="gradient_boosting", n_estimators=10, random_state=42)
        clf.train(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(y)

    def test_svm(self, small_dataset):
        X, y = small_dataset
        clf = ECGClassifier(model_type="svm", random_state=42)
        clf.train(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(y)
