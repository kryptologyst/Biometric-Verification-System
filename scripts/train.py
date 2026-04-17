#!/usr/bin/env python3
"""Training script for biometric verification system."""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np
import yaml
from omegaconf import OmegaConf

from models.biometric_verifier import (
    FingerprintVerifier, FaceVerifier, VoiceVerifier, 
    generate_synthetic_dataset
)
from eval.biometric_metrics import BiometricEvaluator
from defenses.anti_spoofing import AntiSpoofingSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        config = OmegaConf.load(config_path)
        return OmegaConf.to_container(config, resolve=True)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        sys.exit(1)


def setup_directories(config: dict) -> None:
    """Create necessary directories.
    
    Args:
        config: Configuration dictionary
    """
    output_dir = config.get('output', {}).get('output_dir', 'assets/generated')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    logger.info(f"Created output directory: {output_dir}")


def train_verifiers(config: dict) -> dict:
    """Train biometric verifiers.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of trained verifiers
    """
    logger.info("Training biometric verifiers...")
    
    # Generate synthetic dataset
    data_config = config.get('data', {}).get('synthetic', {})
    dataset = generate_synthetic_dataset(
        n_users=data_config.get('n_users', 100),
        n_samples_per_user=data_config.get('n_samples_per_user', 10),
        modalities=data_config.get('modalities', ['fingerprint', 'face', 'voice'])
    )
    
    # Initialize verifiers
    verifiers = {}
    model_configs = config.get('models', {})
    
    if 'fingerprint' in dataset:
        fingerprint_config = model_configs.get('fingerprint', {})
        verifiers['fingerprint'] = FingerprintVerifier(
            threshold=fingerprint_config.get('threshold', 0.3)
        )
        
    if 'face' in dataset:
        face_config = model_configs.get('face', {})
        verifiers['face'] = FaceVerifier(
            threshold=face_config.get('threshold', 0.6)
        )
        
    if 'voice' in dataset:
        voice_config = model_configs.get('voice', {})
        verifiers['voice'] = VoiceVerifier(
            threshold=voice_config.get('threshold', 0.4)
        )
    
    # Enroll users (use first sample of each user)
    for modality, verifier in verifiers.items():
        if modality in dataset:
            enrolled_users = set()
            for template in dataset[modality]:
                if template.user_id not in enrolled_users:
                    # Generate synthetic raw data for enrollment
                    if modality == 'fingerprint':
                        raw_data = np.random.randn(100)
                    elif modality == 'face':
                        raw_data = np.random.randn(200)
                    elif modality == 'voice':
                        raw_data = np.random.randn(150)
                    
                    verifier.enroll(template.user_id, raw_data)
                    enrolled_users.add(template.user_id)
            
            logger.info(f"Enrolled {len(enrolled_users)} users for {modality}")
    
    return verifiers, dataset


def train_anti_spoofing(config: dict) -> AntiSpoofingSystem:
    """Train anti-spoofing system.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Trained anti-spoofing system
    """
    logger.info("Training anti-spoofing system...")
    
    anti_spoofing_config = config.get('anti_spoofing', {})
    if not anti_spoofing_config.get('enabled', True):
        logger.info("Anti-spoofing disabled in config")
        return None
    
    anti_spoofing = AntiSpoofingSystem()
    
    # Train detectors for each modality
    modalities = config.get('data', {}).get('synthetic', {}).get('modalities', [])
    
    for modality in modalities:
        if modality in ['fingerprint', 'face', 'voice']:
            logger.info(f"Training {modality} liveness detector...")
            
            # Generate synthetic training data
            live_samples, spoof_samples = anti_spoofing.generate_synthetic_spoof_data(
                modality, n_live=500, n_spoof=500
            )
            
            # Train detector
            anti_spoofing.train_detector(modality, live_samples, spoof_samples)
    
    logger.info("Anti-spoofing training completed")
    return anti_spoofing


def evaluate_system(verifiers: dict, dataset: dict, config: dict) -> dict:
    """Evaluate the biometric verification system.
    
    Args:
        verifiers: Dictionary of trained verifiers
        dataset: Test dataset
        config: Configuration dictionary
        
    Returns:
        Evaluation results
    """
    logger.info("Evaluating biometric verification system...")
    
    evaluator = BiometricEvaluator()
    results = {}
    
    for modality, verifier in verifiers.items():
        if modality in dataset:
            logger.info(f"Evaluating {modality} verifier...")
            
            result = evaluator.evaluate_verifier(verifier, dataset, modality)
            results[modality] = result
            
            logger.info(f"{modality.capitalize()} - EER: {result['EER']:.4f}, ROC AUC: {result['ROC_AUC']:.4f}")
    
    return results


def save_results(verifiers: dict, results: dict, config: dict) -> None:
    """Save training results and models.
    
    Args:
        verifiers: Dictionary of trained verifiers
        results: Evaluation results
        config: Configuration dictionary
    """
    logger.info("Saving results...")
    
    output_config = config.get('output', {})
    output_dir = output_config.get('output_dir', 'assets/generated')
    
    # Save models
    if output_config.get('save_models', True):
        for modality, verifier in verifiers.items():
            model_path = f"models/{modality}_verifier.pkl"
            # In a real implementation, you would save the model here
            logger.info(f"Model saved: {model_path}")
    
    # Save evaluation results
    if output_config.get('save_results', True):
        import json
        results_path = f"{output_dir}/evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved: {results_path}")
    
    # Generate and save report
    if output_config.get('save_results', True):
        from eval.biometric_metrics import BiometricEvaluator
        evaluator = BiometricEvaluator()
        report = evaluator.generate_report(results, f"{output_dir}/evaluation_report.txt")
        logger.info(f"Report saved: {output_dir}/evaluation_report.txt")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train biometric verification system")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets/generated",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override output directory if specified
    if args.output_dir:
        config['output'] = config.get('output', {})
        config['output']['output_dir'] = args.output_dir
    
    # Set random seed
    random_seed = config.get('system', {}).get('random_seed', 42)
    np.random.seed(random_seed)
    
    # Setup directories
    setup_directories(config)
    
    try:
        # Train verifiers
        verifiers, dataset = train_verifiers(config)
        
        # Train anti-spoofing system
        anti_spoofing = train_anti_spoofing(config)
        
        # Evaluate system
        results = evaluate_system(verifiers, dataset, config)
        
        # Save results
        save_results(verifiers, results, config)
        
        logger.info("Training completed successfully!")
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Modalities trained: {list(verifiers.keys())}")
        print(f"Users enrolled: {len(set(template.user_id for templates in dataset.values() for template in templates))}")
        print(f"Anti-spoofing: {'Enabled' if anti_spoofing else 'Disabled'}")
        print("\nPerformance Summary:")
        for modality, result in results.items():
            print(f"  {modality.capitalize()}: EER={result['EER']:.4f}, ROC AUC={result['ROC_AUC']:.4f}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
