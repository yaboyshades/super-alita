"""
Multi-domain confidence calibration for DeepConf consensus
Implements domain-specific calibration with Platt scaling and isotonic regression
"""

import os
import pickle
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV


@dataclass
class CalibrationProfile:
    """Calibration profile for a specific domain"""

    domain: str
    calibrator_type: str = "platt"  # platt, isotonic, beta
    calibrator: Any = None
    training_samples: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))
    validation_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "calibrator_type": self.calibrator_type,
            "training_samples": self.training_samples,
            "last_updated": self.last_updated.isoformat(),
            "validation_score": self.validation_score,
            "metadata": self.metadata,
        }


class BetaCalibrator:
    """
    Beta distribution-based calibrator for confidence scores
    Particularly effective for bounded confidence values
    """

    def __init__(self):
        self.alpha = 1.0
        self.beta = 1.0
        self.fitted = False

    def fit(self, confidence_scores: list[float], ground_truth: list[bool]):
        """Fit beta distribution parameters using method of moments"""
        if len(confidence_scores) != len(ground_truth):
            raise ValueError(
                "Confidence scores and ground truth must have same length"
            )

        # Convert to numpy arrays
        confidences = np.array(confidence_scores)
        truth = np.array(ground_truth, dtype=float)

        # Calculate empirical mean and variance
        empirical_mean = np.mean(truth)
        empirical_var = np.var(truth)

        # Prevent division by zero
        if empirical_var == 0:
            self.alpha = 1.0
            self.beta = 1.0
        else:
            # Method of moments estimation
            if empirical_mean == 0 or empirical_mean == 1:
                self.alpha = 1.0
                self.beta = 1.0
            else:
                factor = (
                    empirical_mean * (1 - empirical_mean) / empirical_var - 1
                )
                self.alpha = empirical_mean * factor
                self.beta = (1 - empirical_mean) * factor

                # Ensure positive parameters
                self.alpha = max(0.1, self.alpha)
                self.beta = max(0.1, self.beta)

        self.fitted = True

    def predict_proba(self, confidence_scores: list[float]) -> np.ndarray:
        """Transform confidence scores using fitted beta distribution"""
        if not self.fitted:
            raise ValueError("Calibrator must be fitted before prediction")

        confidences = np.array(confidence_scores)

        # Apply beta CDF transformation
        from scipy.stats import beta

        calibrated = beta.cdf(confidences, self.alpha, self.beta)

        # Return in sklearn format (negative class, positive class)
        return np.column_stack([1 - calibrated, calibrated])


class MultiDomainConfidenceCalibrator:
    """
    Multi-domain confidence calibrator with adaptive learning
    Maintains separate calibration profiles for different domains
    """

    def __init__(self, cache_dir: str = "./.confidence_cache"):
        self.cache_dir = cache_dir
        self.profiles: dict[str, CalibrationProfile] = {}
        self.global_profile: CalibrationProfile | None = None

        # Default parameters
        self.min_samples_for_calibration = 10
        self.recalibration_threshold = 0.05  # Retrain if accuracy drops
        self.max_cache_age_days = 30

        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

        # Load existing profiles
        self._load_profiles()

    async def calibrate_confidence(
        self,
        confidence_scores: list[float],
        ground_truth: list[bool] | None = None,
        domain: str | None = None,
    ) -> list[float]:
        """
        Calibrate confidence scores for specified domain
        """
        if not confidence_scores:
            return []

        # Select appropriate profile
        if domain and domain in self.profiles:
            profile = self.profiles[domain]
        elif self.global_profile and self.global_profile.calibrator:
            profile = self.global_profile
        else:
            # No calibration available, return original scores
            return confidence_scores

        # Apply calibration
        if profile.calibrator:
            try:
                scores_array = np.array(confidence_scores).reshape(-1, 1)

                if hasattr(profile.calibrator, "predict_proba"):
                    # Sklearn-style calibrator
                    calibrated_probs = profile.calibrator.predict_proba(
                        scores_array
                    )
                    calibrated_scores = calibrated_probs[:, 1].tolist()
                else:
                    # Custom calibrator
                    calibrated_probs = profile.calibrator.predict_proba(
                        confidence_scores
                    )
                    calibrated_scores = calibrated_probs[:, 1].tolist()

                return calibrated_scores

            except Exception as e:
                print(f"Calibration failed for domain {domain}: {e}")
                return confidence_scores

        return confidence_scores

    async def update_calibration(
        self,
        confidence_scores: list[float],
        ground_truth: list[bool],
        domain: str | None = None,
    ) -> bool:
        """
        Update calibration model with new training data
        """
        if len(confidence_scores) != len(ground_truth):
            raise ValueError(
                "Confidence scores and ground truth must have same length"
            )

        if len(confidence_scores) < self.min_samples_for_calibration:
            print(
                f"Insufficient samples for calibration: {len(confidence_scores)} < {self.min_samples_for_calibration}"
            )
            return False

        try:
            # Determine target domain
            target_domain = domain or "global"

            # Create or update profile
            if target_domain not in self.profiles:
                self.profiles[target_domain] = CalibrationProfile(
                    domain=target_domain,
                    calibrator_type="platt",  # Default to Platt scaling
                )

            profile = self.profiles[target_domain]

            # Choose calibration method based on data characteristics
            calibrator_type = self._select_calibration_method(
                confidence_scores, ground_truth
            )

            # Create and fit calibrator
            if calibrator_type == "platt":
                # Platt scaling (logistic regression)
                base_classifier = DummyClassifier(confidence_scores)
                calibrator = CalibratedClassifierCV(
                    base_classifier, method="sigmoid", cv=3
                )

                # Prepare data for sklearn
                X = np.array(confidence_scores).reshape(-1, 1)
                y = np.array(ground_truth)

                calibrator.fit(X, y)

            elif calibrator_type == "isotonic":
                # Isotonic regression
                base_classifier = DummyClassifier(confidence_scores)
                calibrator = CalibratedClassifierCV(
                    base_classifier, method="isotonic", cv=3
                )

                X = np.array(confidence_scores).reshape(-1, 1)
                y = np.array(ground_truth)

                calibrator.fit(X, y)

            elif calibrator_type == "beta":
                # Beta distribution calibrator
                calibrator = BetaCalibrator()
                calibrator.fit(confidence_scores, ground_truth)

            else:
                raise ValueError(f"Unknown calibrator type: {calibrator_type}")

            # Update profile
            profile.calibrator = calibrator
            profile.calibrator_type = calibrator_type
            profile.training_samples = len(confidence_scores)
            profile.last_updated = datetime.now(UTC)

            # Validate calibration quality
            validation_score = self._validate_calibration(
                calibrator, confidence_scores, ground_truth
            )
            profile.validation_score = validation_score

            # Update metadata
            profile.metadata.update(
                {
                    "data_distribution": {
                        "mean_confidence": np.mean(confidence_scores),
                        "std_confidence": np.std(confidence_scores),
                        "positive_rate": np.mean(ground_truth),
                    },
                    "calibration_quality": {
                        "brier_score": self._calculate_brier_score(
                            confidence_scores, ground_truth
                        ),
                        "ece": self._calculate_expected_calibration_error(
                            confidence_scores, ground_truth
                        ),
                    },
                }
            )

            # Set as global profile if none exists
            if target_domain == "global" or not self.global_profile:
                self.global_profile = profile

            # Save to cache
            self._save_profile(profile)

            return True

        except Exception as e:
            print(f"Failed to update calibration for domain {domain}: {e}")
            return False

    def _select_calibration_method(
        self, confidence_scores: list[float], ground_truth: list[bool]
    ) -> str:
        """
        Select appropriate calibration method based on data characteristics
        """
        n_samples = len(confidence_scores)
        confidence_array = np.array(confidence_scores)

        # Use Platt scaling for small datasets
        if n_samples < 50:
            return "platt"

        # Check if data is well-behaved for isotonic regression
        unique_confidences = len(np.unique(confidence_array))
        if unique_confidences > n_samples * 0.8:  # High diversity
            return "isotonic"

        # Check confidence distribution characteristics
        conf_std = np.std(confidence_array)
        if conf_std < 0.1:  # Low variance, beta might work well
            return "beta"

        # Default to Platt scaling
        return "platt"

    def _validate_calibration(
        self,
        calibrator,
        confidence_scores: list[float],
        ground_truth: list[bool],
    ) -> float:
        """
        Validate calibration quality using cross-validation
        """
        try:
            from sklearn.metrics import brier_score_loss
            from sklearn.model_selection import cross_val_score

            # Create dummy classifier for validation
            X = np.array(confidence_scores).reshape(-1, 1)
            y = np.array(ground_truth)

            # Use Brier score as validation metric (lower is better)
            if hasattr(calibrator, "predict_proba"):
                scores = cross_val_score(
                    calibrator,
                    X,
                    y,
                    scoring=lambda est, X, y: -brier_score_loss(
                        y, est.predict_proba(X)[:, 1]
                    ),
                    cv=min(3, len(confidence_scores) // 5),
                )
                return np.mean(scores)
            else:
                # For custom calibrators, compute single validation
                calibrated_probs = calibrator.predict_proba(confidence_scores)
                return -brier_score_loss(y, calibrated_probs[:, 1])

        except Exception as e:
            print(f"Calibration validation failed: {e}")
            return 0.0

    def _calculate_brier_score(
        self, confidence_scores: list[float], ground_truth: list[bool]
    ) -> float:
        """Calculate Brier score for calibration assessment"""
        confidences = np.array(confidence_scores)
        truth = np.array(ground_truth, dtype=float)
        return np.mean((confidences - truth) ** 2)

    def _calculate_expected_calibration_error(
        self,
        confidence_scores: list[float],
        ground_truth: list[bool],
        n_bins: int = 10,
    ) -> float:
        """Calculate Expected Calibration Error (ECE)"""
        confidences = np.array(confidence_scores)
        truth = np.array(ground_truth, dtype=float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
            # Find predictions in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = truth[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += (
                    np.abs(avg_confidence_in_bin - accuracy_in_bin)
                    * prop_in_bin
                )

        return ece

    def _save_profile(self, profile: CalibrationProfile):
        """Save calibration profile to cache"""
        try:
            cache_file = os.path.join(
                self.cache_dir, f"{profile.domain}_calibration.pkl"
            )
            with open(cache_file, "wb") as f:
                pickle.dump(profile, f)
        except Exception as e:
            print(
                f"Failed to save calibration profile for {profile.domain}: {e}"
            )

    def _load_profiles(self):
        """Load calibration profiles from cache"""
        try:
            if not os.path.exists(self.cache_dir):
                return

            for filename in os.listdir(self.cache_dir):
                if filename.endswith("_calibration.pkl"):
                    cache_file = os.path.join(self.cache_dir, filename)
                    try:
                        with open(cache_file, "rb") as f:
                            profile = pickle.load(f)

                        # Check if profile is not too old
                        age_days = (
                            datetime.now(UTC) - profile.last_updated
                        ).days
                        if age_days <= self.max_cache_age_days:
                            self.profiles[profile.domain] = profile

                            if profile.domain == "global":
                                self.global_profile = profile
                        else:
                            # Remove old cache file
                            os.remove(cache_file)

                    except Exception as e:
                        print(
                            f"Failed to load calibration profile from {filename}: {e}"
                        )

        except Exception as e:
            print(f"Failed to load calibration profiles: {e}")

    def get_calibration_stats(self) -> dict[str, Any]:
        """Get statistics about calibration profiles"""
        stats = {
            "num_profiles": len(self.profiles),
            "domains": list(self.profiles.keys()),
            "global_profile_available": self.global_profile is not None,
            "profiles": {},
        }

        for domain, profile in self.profiles.items():
            stats["profiles"][domain] = profile.to_dict()

        return stats

    def clear_cache(self):
        """Clear all calibration profiles and cache"""
        self.profiles.clear()
        self.global_profile = None

        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith("_calibration.pkl"):
                    os.remove(os.path.join(self.cache_dir, filename))
        except Exception as e:
            print(f"Failed to clear calibration cache: {e}")


class DummyClassifier:
    """
    Dummy classifier that uses confidence scores directly as probabilities
    Used as base for CalibratedClassifierCV
    """

    def __init__(self, confidence_scores: list[float]):
        self.confidence_scores = np.array(confidence_scores)
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        # X contains confidence scores
        confidences = X.flatten()
        return np.column_stack([1 - confidences, confidences])

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)


# Export main classes
__all__ = [
    "MultiDomainConfidenceCalibrator",
    "CalibrationProfile",
    "BetaCalibrator",
]
