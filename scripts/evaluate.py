#!/usr/bin/env python3
"""Evaluation script for biometric verification system."""

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
from eval.biometric_metrics import BiometricEvaluator, create_leaderboard
from defenses.anti_spoofing import AntiSpoofingSystem, evaluate_anti_spoofing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        config = OmegaConf.load(config_path)
        return OmegaConf.to_container(config, resolve=True)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        sys.exit(1)


def load_verifiers(model_paths: dict) -> dict:
    """Load trained verifiers.
    
    Args:
        model_paths: Dictionary mapping modality to model path
        
    Returns:
        Dictionary of loaded verifiers
    """
    verifiers = {}
    
    for modality, path in model_paths.items():
        if modality == 'fingerprint':
            verifiers[modality] = FingerprintVerifier()
        elif modality == 'face':
            verifiers[modality] = FaceVerifier()
        elif modality == 'voice':
            verifiers[modality] = VoiceVerifier()
        
        # In a real implementation, you would load the actual trained model here
        logger.info(f"Loaded {modality} verifier from {path}")
    
    return verifiers


def evaluate_verification_performance(verifiers: dict, dataset: dict) -> dict:
    """Evaluate verification performance.
    
    Args:
        verifiers: Dictionary of verifiers
        dataset: Test dataset
        
    Returns:
        Evaluation results
    """
    logger.info("Evaluating verification performance...")
    
    evaluator = BiometricEvaluator()
    results = {}
    
    for modality, verifier in verifiers.items():
        if modality in dataset:
            logger.info(f"Evaluating {modality} verifier...")
            
            result = evaluator.evaluate_verifier(verifier, dataset, modality)
            results[modality] = result
            
            logger.info(f"{modality.capitalize()} - EER: {result['EER']:.4f}, ROC AUC: {result['ROC_AUC']:.4f}")
    
    return results


def evaluate_anti_spoofing_performance(anti_spoofing: AntiSpoofingSystem, config: dict) -> dict:
    """Evaluate anti-spoofing performance.
    
    Args:
        anti_spoofing: Anti-spoofing system
        config: Configuration dictionary
        
    Returns:
        Anti-spoofing evaluation results
    """
    logger.info("Evaluating anti-spoofing performance...")
    
    modalities = config.get('data', {}).get('synthetic', {}).get('modalities', [])
    results = {}
    
    for modality in modalities:
        if modality in ['fingerprint', 'face', 'voice']:
            logger.info(f"Evaluating {modality} liveness detection...")
            
            # Generate test data
            live_samples, spoof_samples = anti_spoofing.generate_synthetic_spoof_data(
                modality, n_live=200, n_spoof=200
            )
            
            # Evaluate detector
            detector = anti_spoofing.detectors[modality]
            result = evaluate_anti_spoofing(detector, live_samples, spoof_samples)
            results[modality] = result
            
            logger.info(f"{modality.capitalize()} - Live Detection: {result['Live_Detection_Rate']:.3f}, "
                       f"Spoof Detection: {result['Spoof_Detection_Rate']:.3f}")
    
    return results


def generate_plots(results: dict, output_dir: str) -> None:
    """Generate evaluation plots.
    
    Args:
        results: Evaluation results
        output_dir: Output directory for plots
    """
    logger.info("Generating evaluation plots...")
    
    evaluator = BiometricEvaluator()
    
    for modality, metrics in results.items():
        if modality in ['fingerprint', 'face', 'voice']:
            logger.info(f"Generating plots for {modality}...")
            
            # Generate synthetic data for plotting
            dataset = generate_synthetic_dataset(n_users=50, n_samples_per_user=5, modalities=[modality])
            
            # Extract templates
            templates = dataset[modality]
            
            # Generate genuine and impostor scores
            genuine_scores = []
            impostor_scores = []
            
            # Group templates by user
            user_templates = {}
            for template in templates:
                if template.user_id not in user_templates:
                    user_templates[template.user_id] = []
                user_templates[template.user_id].append(template)
            
            # Genuine comparisons
            for user_id, user_template_list in user_templates.items():
                if len(user_template_list) >= 2:
                    template1 = user_template_list[0]
                    for template2 in user_template_list[1:]:
                        similarity = 1 - np.linalg.norm(template1.template - template2.template)
                        genuine_scores.append(similarity)
            
            # Impostor comparisons
            user_ids = list(user_templates.keys())
            for i, user_id1 in enumerate(user_ids):
                for user_id2 in user_ids[i+1:]:
                    template1 = user_templates[user_id1][0]
                    template2 = user_templates[user_id2][0]
                    similarity = 1 - np.linalg.norm(template1.template - template2.template)
                    impostor_scores.append(similarity)
            
            genuine_scores = np.array(genuine_scores)
            impostor_scores = np.array(impostor_scores)
            
            # Generate plots
            plot_dir = f"{output_dir}/plots"
            os.makedirs(plot_dir, exist_ok=True)
            
            # ROC curve
            evaluator.plot_roc_curve(
                genuine_scores, impostor_scores,
                title=f"{modality.capitalize()} ROC Curve",
                save_path=f"{plot_dir}/{modality}_roc_curve.png"
            )
            
            # DET curve
            evaluator.plot_det_curve(
                genuine_scores, impostor_scores,
                title=f"{modality.capitalize()} DET Curve",
                save_path=f"{plot_dir}/{modality}_det_curve.png"
            )
            
            # Score distributions
            evaluator.plot_score_distributions(
                genuine_scores, impostor_scores,
                title=f"{modality.capitalize()} Score Distributions",
                save_path=f"{plot_dir}/{modality}_score_distributions.png"
            )


def save_results(results: dict, anti_spoofing_results: dict, output_dir: str) -> None:
    """Save evaluation results.
    
    Args:
        results: Verification results
        anti_spoofing_results: Anti-spoofing results
        output_dir: Output directory
    """
    logger.info("Saving evaluation results...")
    
    import json
    
    # Save verification results
    verification_path = f"{output_dir}/verification_results.json"
    with open(verification_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save anti-spoofing results
    if anti_spoofing_results:
        anti_spoofing_path = f"{output_dir}/anti_spoofing_results.json"
        with open(anti_spoofing_path, 'w') as f:
            json.dump(anti_spoofing_results, f, indent=2)
    
    # Generate comprehensive report
    from eval.biometric_metrics import BiometricEvaluator
    evaluator = BiometricEvaluator()
    
    report = evaluator.generate_report(results, f"{output_dir}/evaluation_report.txt")
    
    # Generate leaderboard
    leaderboard = create_leaderboard(results)
    with open(f"{output_dir}/leaderboard.txt", 'w') as f:
        f.write(leaderboard)
    
    logger.info(f"Results saved to {output_dir}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate biometric verification system")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models",
        help="Path to trained models directory"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path to test dataset (if None, generates synthetic data)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets/generated",
        help="Output directory for results"
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="Generate evaluation plots"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    random_seed = config.get('system', {}).get('random_seed', 42)
    np.random.seed(random_seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load verifiers
        modalities = config.get('data', {}).get('synthetic', {}).get('modalities', [])
        model_paths = {modality: f"{args.model_path}/{modality}_verifier.pkl" for modality in modalities}
        verifiers = load_verifiers(model_paths)
        
        # Generate or load test data
        if args.test_data:
            # In a real implementation, you would load actual test data here
            logger.info(f"Loading test data from {args.test_data}")
            dataset = generate_synthetic_dataset(n_users=100, n_samples_per_user=10, modalities=modalities)
        else:
            logger.info("Generating synthetic test data...")
            dataset = generate_synthetic_dataset(
                n_users=config.get('data', {}).get('synthetic', {}).get('n_users', 100),
                n_samples_per_user=config.get('data', {}).get('synthetic', {}).get('n_samples_per_user', 10),
                modalities=modalities
            )
        
        # Evaluate verification performance
        results = evaluate_verification_performance(verifiers, dataset)
        
        # Evaluate anti-spoofing performance
        anti_spoofing_results = None
        if config.get('anti_spoofing', {}).get('enabled', True):
            anti_spoofing = AntiSpoofingSystem()
            anti_spoofing_results = evaluate_anti_spoofing_performance(anti_spoofing, config)
        
        # Generate plots
        if args.generate_plots:
            generate_plots(results, args.output_dir)
        
        # Save results
        save_results(results, anti_spoofing_results, args.output_dir)
        
        logger.info("Evaluation completed successfully!")
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Modalities evaluated: {list(results.keys())}")
        print(f"Test users: {len(set(template.user_id for templates in dataset.values() for template in templates))}")
        print(f"Anti-spoofing: {'Evaluated' if anti_spoofing_results else 'Skipped'}")
        print(f"Plots generated: {'Yes' if args.generate_plots else 'No'}")
        print("\nPerformance Summary:")
        for modality, result in results.items():
            print(f"  {modality.capitalize()}: EER={result['EER']:.4f}, ROC AUC={result['ROC_AUC']:.4f}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
