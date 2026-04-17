"""Unit tests for biometric verification system."""

import unittest
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.biometric_verifier import (
    BiometricTemplate, FingerprintVerifier, FaceVerifier, VoiceVerifier,
    MultiModalVerifier, generate_synthetic_dataset
)
from eval.biometric_metrics import BiometricEvaluator
from defenses.anti_spoofing import (
    FingerprintLivenessDetector, FaceLivenessDetector, VoiceLivenessDetector,
    AntiSpoofingSystem
)


class TestBiometricTemplate(unittest.TestCase):
    """Test cases for BiometricTemplate class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.template = BiometricTemplate(
            user_id="test_user",
            modality="fingerprint",
            template=np.random.randn(128),
            quality_score=0.95
        )
    
    def test_template_creation(self):
        """Test template creation."""
        self.assertEqual(self.template.user_id, "test_user")
        self.assertEqual(self.template.modality, "fingerprint")
        self.assertEqual(self.template.quality_score, 0.95)
        self.assertEqual(len(self.template.template), 128)
    
    def test_template_to_dict(self):
        """Test template to dictionary conversion."""
        template_dict = self.template.to_dict()
        self.assertIn('user_id', template_dict)
        self.assertIn('modality', template_dict)
        self.assertIn('template', template_dict)
        self.assertIn('quality_score', template_dict)


class TestFingerprintVerifier(unittest.TestCase):
    """Test cases for FingerprintVerifier class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.verifier = FingerprintVerifier(threshold=0.3)
        self.test_data = np.random.randn(100)
    
    def test_feature_extraction(self):
        """Test feature extraction."""
        features = self.verifier.extract_features(self.test_data)
        self.assertEqual(len(features), 128)
        self.assertAlmostEqual(np.linalg.norm(features), 1.0, places=5)
    
    def test_similarity_computation(self):
        """Test similarity computation."""
        template1 = np.random.randn(128)
        template2 = template1 + np.random.normal(0, 0.1, 128)
        
        similarity = self.verifier.compute_similarity(template1, template2)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
    
    def test_enrollment(self):
        """Test user enrollment."""
        success = self.verifier.enroll("test_user", self.test_data)
        self.assertTrue(success)
        self.assertIn("test_user", self.verifier.templates)
    
    def test_verification(self):
        """Test user verification."""
        # Enroll user first
        self.verifier.enroll("test_user", self.test_data)
        
        # Test verification
        is_verified, score = self.verifier.verify("test_user", self.test_data)
        self.assertIsInstance(is_verified, bool)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestFaceVerifier(unittest.TestCase):
    """Test cases for FaceVerifier class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.verifier = FaceVerifier(threshold=0.6)
        self.test_data = np.random.randn(200)
    
    def test_feature_extraction(self):
        """Test feature extraction."""
        features = self.verifier.extract_features(self.test_data)
        self.assertEqual(len(features), 512)
        self.assertAlmostEqual(np.linalg.norm(features), 1.0, places=5)
    
    def test_enrollment_and_verification(self):
        """Test enrollment and verification."""
        # Enroll user
        success = self.verifier.enroll("test_user", self.test_data)
        self.assertTrue(success)
        
        # Verify user
        is_verified, score = self.verifier.verify("test_user", self.test_data)
        self.assertIsInstance(is_verified, bool)
        self.assertIsInstance(score, float)


class TestVoiceVerifier(unittest.TestCase):
    """Test cases for VoiceVerifier class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.verifier = VoiceVerifier(threshold=0.4)
        self.test_data = np.random.randn(150)
    
    def test_feature_extraction(self):
        """Test feature extraction."""
        features = self.verifier.extract_features(self.test_data)
        self.assertEqual(len(features), 256)
        self.assertAlmostEqual(np.linalg.norm(features), 1.0, places=5)
    
    def test_enrollment_and_verification(self):
        """Test enrollment and verification."""
        # Enroll user
        success = self.verifier.enroll("test_user", self.test_data)
        self.assertTrue(success)
        
        # Verify user
        is_verified, score = self.verifier.verify("test_user", self.test_data)
        self.assertIsInstance(is_verified, bool)
        self.assertIsInstance(score, float)


class TestMultiModalVerifier(unittest.TestCase):
    """Test cases for MultiModalVerifier class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.verifier = MultiModalVerifier()
        self.test_data = {
            'fingerprint': np.random.randn(100),
            'face': np.random.randn(200),
            'voice': np.random.randn(150)
        }
    
    def test_enrollment(self):
        """Test multi-modal enrollment."""
        for modality, data in self.test_data.items():
            success = self.verifier.enroll("test_user", modality, data)
            self.assertTrue(success)
    
    def test_verification(self):
        """Test multi-modal verification."""
        # Enroll user first
        for modality, data in self.test_data.items():
            self.verifier.enroll("test_user", modality, data)
        
        # Test single modality verification
        is_verified, score = self.verifier.verify("test_user", "fingerprint", self.test_data['fingerprint'])
        self.assertIsInstance(is_verified, bool)
        self.assertIsInstance(score, float)
        
        # Test multi-modal verification
        is_verified, score = self.verifier.verify_multimodal("test_user", self.test_data)
        self.assertIsInstance(is_verified, bool)
        self.assertIsInstance(score, float)


class TestBiometricEvaluator(unittest.TestCase):
    """Test cases for BiometricEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = BiometricEvaluator()
        self.genuine_scores = np.random.uniform(0.6, 1.0, 100)
        self.impostor_scores = np.random.uniform(0.0, 0.4, 100)
    
    def test_eer_computation(self):
        """Test EER computation."""
        eer = self.evaluator.compute_eer(self.genuine_scores, self.impostor_scores)
        self.assertGreaterEqual(eer, 0.0)
        self.assertLessEqual(eer, 1.0)
    
    def test_mindcf_computation(self):
        """Test minDCF computation."""
        min_dcf = self.evaluator.compute_mindcf(self.genuine_scores, self.impostor_scores)
        self.assertGreaterEqual(min_dcf, 0.0)
    
    def test_far_frr_computation(self):
        """Test FAR and FRR computation."""
        threshold = 0.5
        far, frr = self.evaluator.compute_far_frr(self.genuine_scores, self.impostor_scores, threshold)
        self.assertGreaterEqual(far, 0.0)
        self.assertLessEqual(far, 1.0)
        self.assertGreaterEqual(frr, 0.0)
        self.assertLessEqual(frr, 1.0)


class TestLivenessDetectors(unittest.TestCase):
    """Test cases for liveness detectors."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fingerprint_detector = FingerprintLivenessDetector()
        self.face_detector = FaceLivenessDetector()
        self.voice_detector = VoiceLivenessDetector()
        
        self.test_data = np.random.randn(100)
    
    def test_fingerprint_liveness_detection(self):
        """Test fingerprint liveness detection."""
        features = self.fingerprint_detector.extract_liveness_features(self.test_data)
        self.assertEqual(len(features), 6)  # Number of liveness features
        
        is_live, confidence = self.fingerprint_detector.predict_liveness(self.test_data)
        self.assertIsInstance(is_live, bool)
        self.assertIsInstance(confidence, float)
    
    def test_face_liveness_detection(self):
        """Test face liveness detection."""
        features = self.face_detector.extract_liveness_features(self.test_data)
        self.assertEqual(len(features), 6)
        
        is_live, confidence = self.face_detector.predict_liveness(self.test_data)
        self.assertIsInstance(is_live, bool)
        self.assertIsInstance(confidence, float)
    
    def test_voice_liveness_detection(self):
        """Test voice liveness detection."""
        features = self.voice_detector.extract_liveness_features(self.test_data)
        self.assertEqual(len(features), 6)
        
        is_live, confidence = self.voice_detector.predict_liveness(self.test_data)
        self.assertIsInstance(is_live, bool)
        self.assertIsInstance(confidence, float)


class TestAntiSpoofingSystem(unittest.TestCase):
    """Test cases for AntiSpoofingSystem class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.anti_spoofing = AntiSpoofingSystem()
    
    def test_liveness_check(self):
        """Test liveness check."""
        test_data = np.random.randn(100)
        
        is_live, confidence = self.anti_spoofing.check_liveness("fingerprint", test_data)
        self.assertIsInstance(is_live, bool)
        self.assertIsInstance(confidence, float)
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        live_samples, spoof_samples = self.anti_spoofing.generate_synthetic_spoof_data(
            "fingerprint", n_live=10, n_spoof=10
        )
        
        self.assertEqual(len(live_samples), 10)
        self.assertEqual(len(spoof_samples), 10)
        
        for sample in live_samples + spoof_samples:
            self.assertIsInstance(sample, np.ndarray)


class TestSyntheticDataGeneration(unittest.TestCase):
    """Test cases for synthetic data generation."""
    
    def test_dataset_generation(self):
        """Test synthetic dataset generation."""
        dataset = generate_synthetic_dataset(n_users=10, n_samples_per_user=5)
        
        self.assertIn('fingerprint', dataset)
        self.assertIn('face', dataset)
        self.assertIn('voice', dataset)
        
        # Check that we have the right number of templates
        for modality, templates in dataset.items():
            self.assertEqual(len(templates), 10 * 5)  # n_users * n_samples_per_user
            
            # Check template properties
            for template in templates:
                self.assertIsInstance(template, BiometricTemplate)
                self.assertIn(template.user_id, [f"user_{i:03d}" for i in range(10)])


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
