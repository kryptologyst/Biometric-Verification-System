"""Biometric verification system package."""

__version__ = "1.0.0"
__author__ = "AI Research Team"
__email__ = "research@example.com"

from .models.biometric_verifier import (
    BiometricTemplate,
    FingerprintVerifier,
    FaceVerifier,
    VoiceVerifier,
    MultiModalVerifier,
    generate_synthetic_dataset
)

from .eval.biometric_metrics import (
    BiometricEvaluator,
    create_leaderboard
)

from .defenses.anti_spoofing import (
    FingerprintLivenessDetector,
    FaceLivenessDetector,
    VoiceLivenessDetector,
    AntiSpoofingSystem
)

__all__ = [
    "BiometricTemplate",
    "FingerprintVerifier",
    "FaceVerifier", 
    "VoiceVerifier",
    "MultiModalVerifier",
    "generate_synthetic_dataset",
    "BiometricEvaluator",
    "create_leaderboard",
    "FingerprintLivenessDetector",
    "FaceLivenessDetector",
    "VoiceLivenessDetector",
    "AntiSpoofingSystem"
]
