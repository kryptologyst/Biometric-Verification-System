"""Anti-spoofing and liveness detection for biometric systems."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class LivenessDetector(ABC):
    """Abstract base class for liveness detection."""
    
    def __init__(self):
        """Initialize liveness detector."""
        self.is_trained = False
        
    @abstractmethod
    def extract_liveness_features(self, raw_data: np.ndarray) -> np.ndarray:
        """Extract features for liveness detection."""
        pass
        
    @abstractmethod
    def predict_liveness(self, raw_data: np.ndarray) -> Tuple[bool, float]:
        """Predict if the biometric sample is live.
        
        Returns:
            Tuple of (is_live, confidence_score)
        """
        pass


class FingerprintLivenessDetector(LivenessDetector):
    """Liveness detection for fingerprint biometrics."""
    
    def __init__(self):
        """Initialize fingerprint liveness detector."""
        super().__init__()
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = [
            'texture_variance', 'ridge_frequency', 'ridge_orientation',
            'sweat_pore_density', 'elasticity_score', 'conductivity_score'
        ]
        
    def extract_liveness_features(self, raw_data: np.ndarray) -> np.ndarray:
        """Extract liveness features from fingerprint data.
        
        In a real system, these would be computed from actual fingerprint images
        using computer vision techniques.
        """
        features = np.zeros(len(self.feature_names))
        
        # Simulate feature extraction
        if len(raw_data.shape) == 1:
            # Flattened image data
            features[0] = np.var(raw_data)  # Texture variance
            features[1] = np.mean(np.abs(np.diff(raw_data)))  # Ridge frequency proxy
            features[2] = np.std(raw_data)  # Ridge orientation proxy
        else:
            # Image-like data
            features[0] = np.var(raw_data)
            features[1] = np.mean(np.abs(np.diff(raw_data.flatten())))
            features[2] = np.std(raw_data)
            
        # Simulate additional liveness features
        features[3] = np.random.uniform(0.1, 0.9)  # Sweat pore density
        features[4] = np.random.uniform(0.2, 1.0)  # Elasticity score
        features[5] = np.random.uniform(0.1, 0.8)  # Conductivity score
        
        return features
        
    def train(self, live_samples: List[np.ndarray], spoof_samples: List[np.ndarray]) -> None:
        """Train the liveness detector.
        
        Args:
            live_samples: List of live biometric samples
            spoof_samples: List of spoofed biometric samples
        """
        logger.info("Training fingerprint liveness detector...")
        
        # Extract features
        live_features = [self.extract_liveness_features(sample) for sample in live_samples]
        spoof_features = [self.extract_liveness_features(sample) for sample in spoof_samples]
        
        # Combine features and labels
        X = np.vstack([live_features, spoof_features])
        y = np.hstack([np.ones(len(live_features)), np.zeros(len(spoof_features))])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train classifier
        self.classifier.fit(X_scaled, y)
        self.is_trained = True
        
        logger.info(f"Liveness detector trained on {len(X)} samples")
        
    def predict_liveness(self, raw_data: np.ndarray) -> Tuple[bool, float]:
        """Predict liveness of fingerprint sample.
        
        Args:
            raw_data: Raw fingerprint data
            
        Returns:
            Tuple of (is_live, confidence_score)
        """
        if not self.is_trained:
            logger.warning("Liveness detector not trained, returning default prediction")
            return True, 0.5
            
        features = self.extract_liveness_features(raw_data)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get prediction and probability
        prediction = self.classifier.predict(features_scaled)[0]
        confidence = self.classifier.predict_proba(features_scaled)[0].max()
        
        is_live = bool(prediction)
        
        return is_live, confidence


class FaceLivenessDetector(LivenessDetector):
    """Liveness detection for face biometrics."""
    
    def __init__(self):
        """Initialize face liveness detector."""
        super().__init__()
        self.classifier = SVC(probability=True, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = [
            'eye_blink_rate', 'head_movement', 'facial_expression_change',
            'skin_texture', 'depth_variation', 'reflection_pattern'
        ]
        
    def extract_liveness_features(self, raw_data: np.ndarray) -> np.ndarray:
        """Extract liveness features from face data."""
        features = np.zeros(len(self.feature_names))
        
        # Simulate feature extraction
        if len(raw_data.shape) == 1:
            features[0] = np.random.uniform(0.1, 0.8)  # Eye blink rate
            features[1] = np.random.uniform(0.2, 1.0)  # Head movement
            features[2] = np.random.uniform(0.1, 0.9)  # Expression change
        else:
            features[0] = np.random.uniform(0.1, 0.8)
            features[1] = np.random.uniform(0.2, 1.0)
            features[2] = np.random.uniform(0.1, 0.9)
            
        # Additional liveness features
        features[3] = np.random.uniform(0.3, 1.0)  # Skin texture
        features[4] = np.random.uniform(0.2, 0.9)  # Depth variation
        features[5] = np.random.uniform(0.1, 0.7)  # Reflection pattern
        
        return features
        
    def train(self, live_samples: List[np.ndarray], spoof_samples: List[np.ndarray]) -> None:
        """Train the face liveness detector."""
        logger.info("Training face liveness detector...")
        
        live_features = [self.extract_liveness_features(sample) for sample in live_samples]
        spoof_features = [self.extract_liveness_features(sample) for sample in spoof_samples]
        
        X = np.vstack([live_features, spoof_features])
        y = np.hstack([np.ones(len(live_features)), np.zeros(len(spoof_features))])
        
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, y)
        self.is_trained = True
        
        logger.info(f"Face liveness detector trained on {len(X)} samples")
        
    def predict_liveness(self, raw_data: np.ndarray) -> Tuple[bool, float]:
        """Predict liveness of face sample."""
        if not self.is_trained:
            return True, 0.5
            
        features = self.extract_liveness_features(raw_data)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        prediction = self.classifier.predict(features_scaled)[0]
        confidence = self.classifier.predict_proba(features_scaled)[0].max()
        
        is_live = bool(prediction)
        
        return is_live, confidence


class VoiceLivenessDetector(LivenessDetector):
    """Liveness detection for voice biometrics."""
    
    def __init__(self):
        """Initialize voice liveness detector."""
        super().__init__()
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = [
            'spectral_centroid', 'spectral_rolloff', 'zero_crossing_rate',
            'mfcc_variance', 'pitch_variation', 'energy_distribution'
        ]
        
    def extract_liveness_features(self, raw_data: np.ndarray) -> np.ndarray:
        """Extract liveness features from voice data."""
        features = np.zeros(len(self.feature_names))
        
        # Simulate audio feature extraction
        if len(raw_data.shape) == 1:
            # Audio signal
            features[0] = np.mean(raw_data)  # Spectral centroid proxy
            features[1] = np.std(raw_data)   # Spectral rolloff proxy
            features[2] = np.sum(np.diff(np.sign(raw_data)) != 0) / len(raw_data)  # ZCR
        else:
            # Spectrogram-like data
            features[0] = np.mean(raw_data)
            features[1] = np.std(raw_data)
            features[2] = np.random.uniform(0.1, 0.5)
            
        # Additional voice liveness features
        features[3] = np.random.uniform(0.2, 1.0)  # MFCC variance
        features[4] = np.random.uniform(0.1, 0.8)  # Pitch variation
        features[5] = np.random.uniform(0.3, 1.0)  # Energy distribution
        
        return features
        
    def train(self, live_samples: List[np.ndarray], spoof_samples: List[np.ndarray]) -> None:
        """Train the voice liveness detector."""
        logger.info("Training voice liveness detector...")
        
        live_features = [self.extract_liveness_features(sample) for sample in live_samples]
        spoof_features = [self.extract_liveness_features(sample) for sample in spoof_samples]
        
        X = np.vstack([live_features, spoof_features])
        y = np.hstack([np.ones(len(live_features)), np.zeros(len(spoof_features))])
        
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, y)
        self.is_trained = True
        
        logger.info(f"Voice liveness detector trained on {len(X)} samples")
        
    def predict_liveness(self, raw_data: np.ndarray) -> Tuple[bool, float]:
        """Predict liveness of voice sample."""
        if not self.is_trained:
            return True, 0.5
            
        features = self.extract_liveness_features(raw_data)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        prediction = self.classifier.predict(features_scaled)[0]
        confidence = self.classifier.predict_proba(features_scaled)[0].max()
        
        is_live = bool(prediction)
        
        return is_live, confidence


class AntiSpoofingSystem:
    """Comprehensive anti-spoofing system for multiple biometric modalities."""
    
    def __init__(self):
        """Initialize anti-spoofing system."""
        self.detectors = {
            'fingerprint': FingerprintLivenessDetector(),
            'face': FaceLivenessDetector(),
            'voice': VoiceLivenessDetector()
        }
        self.liveness_threshold = 0.7
        
    def train_detector(self, modality: str, live_samples: List[np.ndarray], spoof_samples: List[np.ndarray]) -> None:
        """Train liveness detector for specific modality.
        
        Args:
            modality: Biometric modality
            live_samples: List of live samples
            spoof_samples: List of spoofed samples
        """
        if modality not in self.detectors:
            logger.error(f"Unknown modality: {modality}")
            return
            
        self.detectors[modality].train(live_samples, spoof_samples)
        
    def check_liveness(self, modality: str, raw_data: np.ndarray) -> Tuple[bool, float]:
        """Check liveness of biometric sample.
        
        Args:
            modality: Biometric modality
            raw_data: Raw biometric data
            
        Returns:
            Tuple of (is_live, confidence_score)
        """
        if modality not in self.detectors:
            logger.error(f"Unknown modality: {modality}")
            return False, 0.0
            
        is_live, confidence = self.detectors[modality].predict_liveness(raw_data)
        
        # Apply threshold
        is_live_thresholded = confidence >= self.liveness_threshold
        
        logger.info(f"Liveness check for {modality}: {is_live_thresholded} (confidence: {confidence:.3f})")
        
        return is_live_thresholded, confidence
        
    def set_liveness_threshold(self, threshold: float) -> None:
        """Set the liveness detection threshold."""
        self.liveness_threshold = threshold
        logger.info(f"Liveness threshold set to {threshold}")
        
    def generate_synthetic_spoof_data(
        self, 
        modality: str, 
        n_live: int = 1000, 
        n_spoof: int = 1000
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Generate synthetic live and spoof data for training.
        
        Args:
            modality: Biometric modality
            n_live: Number of live samples
            n_spoof: Number of spoof samples
            
        Returns:
            Tuple of (live_samples, spoof_samples)
        """
        logger.info(f"Generating synthetic {modality} liveness data...")
        
        live_samples = []
        spoof_samples = []
        
        # Generate live samples (higher quality, more natural variation)
        for _ in range(n_live):
            if modality == 'fingerprint':
                sample = np.random.randn(100) + np.random.normal(0, 0.1, 100)
            elif modality == 'face':
                sample = np.random.randn(200) + np.random.normal(0, 0.1, 200)
            elif modality == 'voice':
                sample = np.random.randn(150) + np.random.normal(0, 0.1, 150)
            else:
                sample = np.random.randn(100)
                
            live_samples.append(sample)
            
        # Generate spoof samples (lower quality, artificial patterns)
        for _ in range(n_spoof):
            if modality == 'fingerprint':
                # Spoof: more uniform, less natural variation
                sample = np.random.uniform(-1, 1, 100) + np.random.normal(0, 0.3, 100)
            elif modality == 'face':
                # Spoof: more static, less natural
                sample = np.random.uniform(-0.5, 0.5, 200) + np.random.normal(0, 0.2, 200)
            elif modality == 'voice':
                # Spoof: more artificial, less natural variation
                sample = np.random.uniform(-0.8, 0.8, 150) + np.random.normal(0, 0.25, 150)
            else:
                sample = np.random.uniform(-1, 1, 100)
                
            spoof_samples.append(sample)
            
        logger.info(f"Generated {len(live_samples)} live and {len(spoof_samples)} spoof samples")
        
        return live_samples, spoof_samples


def evaluate_anti_spoofing(
    detector: LivenessDetector,
    live_samples: List[np.ndarray],
    spoof_samples: List[np.ndarray]
) -> Dict[str, float]:
    """Evaluate anti-spoofing performance.
    
    Args:
        detector: Liveness detector to evaluate
        live_samples: List of live samples
        spoof_samples: List of spoofed samples
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating anti-spoofing performance...")
    
    # Test on live samples
    live_predictions = []
    live_confidences = []
    
    for sample in live_samples:
        is_live, confidence = detector.predict_liveness(sample)
        live_predictions.append(is_live)
        live_confidences.append(confidence)
        
    # Test on spoof samples
    spoof_predictions = []
    spoof_confidences = []
    
    for sample in spoof_samples:
        is_live, confidence = detector.predict_liveness(sample)
        spoof_predictions.append(is_live)
        spoof_confidences.append(confidence)
        
    # Compute metrics
    live_accuracy = np.mean(live_predictions)  # True positive rate
    spoof_detection_rate = 1 - np.mean(spoof_predictions)  # True negative rate
    
    # False positive rate (live samples classified as spoof)
    fpr = 1 - live_accuracy
    
    # False negative rate (spoof samples classified as live)
    fnr = np.mean(spoof_predictions)
    
    # Overall accuracy
    all_predictions = live_predictions + spoof_predictions
    all_labels = [True] * len(live_samples) + [False] * len(spoof_samples)
    overall_accuracy = np.mean([p == l for p, l in zip(all_predictions, all_labels)])
    
    metrics = {
        'Live_Detection_Rate': live_accuracy,
        'Spoof_Detection_Rate': spoof_detection_rate,
        'False_Positive_Rate': fpr,
        'False_Negative_Rate': fnr,
        'Overall_Accuracy': overall_accuracy,
        'Live_Confidence_Mean': np.mean(live_confidences),
        'Spoof_Confidence_Mean': np.mean(spoof_confidences)
    }
    
    logger.info(f"Anti-spoofing evaluation completed:")
    logger.info(f"  Live Detection Rate: {live_accuracy:.3f}")
    logger.info(f"  Spoof Detection Rate: {spoof_detection_rate:.3f}")
    logger.info(f"  Overall Accuracy: {overall_accuracy:.3f}")
    
    return metrics
