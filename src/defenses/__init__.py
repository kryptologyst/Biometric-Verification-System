"""Defenses package for biometric verification system."""

from .anti_spoofing import (
    LivenessDetector,
    FingerprintLivenessDetector,
    FaceLivenessDetector,
    VoiceLivenessDetector,
    AntiSpoofingSystem,
    evaluate_anti_spoofing
)

__all__ = [
    "LivenessDetector",
    "FingerprintLivenessDetector",
    "FaceLivenessDetector",
    "VoiceLivenessDetector",
    "AntiSpoofingSystem",
    "evaluate_anti_spoofing"
]
