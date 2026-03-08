"""ML model for ECG-based cardiac disease prediction.

Provides :class:`ECGClassifier`, a wrapper around scikit-learn that handles
feature extraction, training, evaluation, and prediction on ECG signals.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

from ecg_analysis.preprocessing import preprocess_ecg
from ecg_analysis.feature_extraction import extract_features


SUPPORTED_MODELS = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "svm": SVC,
}

DISEASE_LABELS = {
    0: "Normal",
    1: "Arrhythmia",
    2: "Myocardial Infarction",
    3: "Atrial Fibrillation",
    4: "Bradycardia",
}


class ECGClassifier:
    """Train and use an ML classifier for ECG-based disease prediction.

    Parameters
    ----------
    model_type : str
        One of ``"random_forest"``, ``"gradient_boosting"``, or ``"svm"``.
    random_state : int
        Random seed for reproducibility.
    **model_params
        Additional keyword arguments forwarded to the scikit-learn estimator.

    Examples
    --------
    >>> clf = ECGClassifier(model_type="random_forest", n_estimators=100)
    >>> clf.train(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    """

    def __init__(self, model_type="random_forest", random_state=42, **model_params):
        if model_type not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type '{model_type}'. "
                f"Choose from {list(SUPPORTED_MODELS.keys())}."
            )
        self.model_type = model_type
        self.random_state = random_state
        self.scaler = StandardScaler()

        params = {"random_state": random_state, **model_params}
        if model_type == "svm":
            params.setdefault("probability", True)
        self.model = SUPPORTED_MODELS[model_type](**params)

    def prepare_features(self, ecg_signals, fs=360):
        """Extract features from a collection of ECG signals.

        Parameters
        ----------
        ecg_signals : list of array-like
            Each element is a single ECG recording.
        fs : int
            Sampling frequency in Hz.

        Returns
        -------
        pandas.DataFrame
            Feature matrix with one row per signal.
        """
        feature_dicts = []
        for signal in ecg_signals:
            preprocessed = preprocess_ecg(signal, fs=fs)
            features = extract_features(
                preprocessed["normalized"], fs=fs, r_peaks=preprocessed["r_peaks"]
            )
            feature_dicts.append(features)
        return pd.DataFrame(feature_dicts)

    def train(self, X, y):
        """Train the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        """Predict disease labels for new feature vectors.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        numpy.ndarray
            Predicted labels.
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        numpy.ndarray
            Predicted probabilities for each class.
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def evaluate(self, X, y):
        """Evaluate the classifier on a test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        dict
            Dictionary with ``accuracy``, ``classification_report``, and
            ``confusion_matrix``.
        """
        predictions = self.predict(X)
        return {
            "accuracy": accuracy_score(y, predictions),
            "classification_report": classification_report(y, predictions, zero_division=0),
            "confusion_matrix": confusion_matrix(y, predictions),
        }

    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Target labels.
        cv : int
            Number of cross-validation folds.

        Returns
        -------
        dict
            Dictionary with ``mean_accuracy`` and ``std_accuracy``.
        """
        X_scaled = self.scaler.fit_transform(X)
        scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring="accuracy")
        return {
            "mean_accuracy": np.mean(scores),
            "std_accuracy": np.std(scores),
        }

    def save_model(self, filepath):
        """Persist the trained model and scaler to disk.

        Parameters
        ----------
        filepath : str
            Destination file path (e.g. ``"ecg_model.joblib"``).
        """
        joblib.dump({"model": self.model, "scaler": self.scaler}, filepath)

    @classmethod
    def load_model(cls, filepath, model_type="random_forest"):
        """Load a previously saved model from disk.

        Parameters
        ----------
        filepath : str
            Path to the saved model file.
        model_type : str
            The model type string used when the model was created.

        Returns
        -------
        ECGClassifier
            Classifier with restored model and scaler.
        """
        data = joblib.load(filepath)
        instance = cls(model_type=model_type)
        instance.model = data["model"]
        instance.scaler = data["scaler"]
        return instance

    def predict_disease(self, X):
        """Predict diseases and return human-readable label names.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        list of str
            Disease names for each sample.
        """
        predictions = self.predict(X)
        return [DISEASE_LABELS.get(p, f"Unknown ({p})") for p in predictions]
