"""Biometric-specific evaluation metrics and analysis tools."""

import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)

logger = logging.getLogger(__name__)


class BiometricEvaluator:
    """Comprehensive evaluation for biometric verification systems."""
    
    def __init__(self):
        """Initialize biometric evaluator."""
        self.results = {}
        
    def compute_eer(self, genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> float:
        """Compute Equal Error Rate (EER).
        
        Args:
            genuine_scores: Similarity scores for genuine comparisons
            impostor_scores: Similarity scores for impostor comparisons
            
        Returns:
            EER value
        """
        # Combine scores and labels
        all_scores = np.concatenate([genuine_scores, impostor_scores])
        labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(labels, all_scores)
        
        # Find EER point (where FPR = 1 - TPR)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = fpr[eer_idx]
        
        return eer
        
    def compute_mindcf(
        self, 
        genuine_scores: np.ndarray, 
        impostor_scores: np.ndarray,
        c_miss: float = 1.0,
        c_fa: float = 1.0,
        p_target: float = 0.01
    ) -> float:
        """Compute minimum Detection Cost Function (minDCF).
        
        Args:
            genuine_scores: Similarity scores for genuine comparisons
            impostor_scores: Similarity scores for impostor comparisons
            c_miss: Cost of miss (false rejection)
            c_fa: Cost of false alarm (false acceptance)
            p_target: Prior probability of target
            
        Returns:
            minDCF value
        """
        # Combine scores and labels
        all_scores = np.concatenate([genuine_scores, impostor_scores])
        labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(labels, all_scores)
        
        # Compute DCF for each threshold
        fnr = 1 - tpr
        dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
        
        # Find minimum DCF
        min_dcf = np.min(dcf)
        
        return min_dcf
        
    def compute_far_frr(
        self, 
        genuine_scores: np.ndarray, 
        impostor_scores: np.ndarray,
        threshold: float
    ) -> Tuple[float, float]:
        """Compute False Acceptance Rate (FAR) and False Rejection Rate (FRR).
        
        Args:
            genuine_scores: Similarity scores for genuine comparisons
            impostor_scores: Similarity scores for impostor comparisons
            threshold: Decision threshold
            
        Returns:
            Tuple of (FAR, FRR)
        """
        # FAR: proportion of impostors accepted
        far = np.mean(impostor_scores >= threshold)
        
        # FRR: proportion of genuines rejected
        frr = np.mean(genuine_scores < threshold)
        
        return far, frr
        
    def plot_roc_curve(
        self, 
        genuine_scores: np.ndarray, 
        impostor_scores: np.ndarray,
        title: str = "ROC Curve",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot ROC curve for biometric verification.
        
        Args:
            genuine_scores: Similarity scores for genuine comparisons
            impostor_scores: Similarity scores for impostor comparisons
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Combine scores and labels
        all_scores = np.concatenate([genuine_scores, impostor_scores])
        labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(labels, all_scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        # Mark EER point
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = fpr[eer_idx]
        ax.plot(eer, 1-eer, 'ro', markersize=8, label=f'EER = {eer:.3f}')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (FAR)')
        ax.set_ylabel('True Positive Rate (1-FRR)')
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_det_curve(
        self, 
        genuine_scores: np.ndarray, 
        impostor_scores: np.ndarray,
        title: str = "DET Curve",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot Detection Error Tradeoff (DET) curve.
        
        Args:
            genuine_scores: Similarity scores for genuine comparisons
            impostor_scores: Similarity scores for impostor comparisons
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Combine scores and labels
        all_scores = np.concatenate([genuine_scores, impostor_scores])
        labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(labels, all_scores)
        
        # Convert to DET curve (log scale)
        fnr = 1 - tpr
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.semilogx(fpr, fnr, 'b-', lw=2, label='DET Curve')
        
        # Mark EER point
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = fpr[eer_idx]
        ax.semilogx(eer, eer, 'ro', markersize=8, label=f'EER = {eer:.3f}')
        
        ax.set_xlim([1e-4, 1])
        ax.set_ylim([1e-4, 1])
        ax.set_xlabel('False Positive Rate (FAR)')
        ax.set_ylabel('False Negative Rate (FRR)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_score_distributions(
        self, 
        genuine_scores: np.ndarray, 
        impostor_scores: np.ndarray,
        title: str = "Score Distributions",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot score distributions for genuine and impostor comparisons.
        
        Args:
            genuine_scores: Similarity scores for genuine comparisons
            impostor_scores: Similarity scores for impostor comparisons
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histograms
        ax.hist(genuine_scores, bins=50, alpha=0.7, label='Genuine', color='green', density=True)
        ax.hist(impostor_scores, bins=50, alpha=0.7, label='Impostor', color='red', density=True)
        
        # Add vertical lines for statistics
        ax.axvline(np.mean(genuine_scores), color='green', linestyle='--', alpha=0.8, label=f'Genuine Mean: {np.mean(genuine_scores):.3f}')
        ax.axvline(np.mean(impostor_scores), color='red', linestyle='--', alpha=0.8, label=f'Impostor Mean: {np.mean(impostor_scores):.3f}')
        
        ax.set_xlabel('Similarity Score')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def evaluate_verifier(
        self, 
        verifier, 
        test_data: Dict[str, List],
        modality: str
    ) -> Dict[str, float]:
        """Comprehensive evaluation of a biometric verifier.
        
        Args:
            verifier: Biometric verifier instance
            test_data: Test dataset
            modality: Biometric modality
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {modality} verifier...")
        
        # Extract templates for this modality
        templates = test_data[modality]
        
        # Generate genuine and impostor comparisons
        genuine_scores = []
        impostor_scores = []
        
        # Group templates by user
        user_templates = {}
        for template in templates:
            if template.user_id not in user_templates:
                user_templates[template.user_id] = []
            user_templates[template.user_id].append(template)
        
        # Genuine comparisons (same user, different samples)
        for user_id, user_template_list in user_templates.items():
            if len(user_template_list) >= 2:
                # Compare first template with others
                template1 = user_template_list[0]
                for template2 in user_template_list[1:]:
                    similarity = verifier.compute_similarity(template1.template, template2.template)
                    genuine_scores.append(similarity)
        
        # Impostor comparisons (different users)
        user_ids = list(user_templates.keys())
        for i, user_id1 in enumerate(user_ids):
            for user_id2 in user_ids[i+1:]:
                template1 = user_templates[user_id1][0]
                template2 = user_templates[user_id2][0]
                similarity = verifier.compute_similarity(template1.template, template2.template)
                impostor_scores.append(similarity)
        
        genuine_scores = np.array(genuine_scores)
        impostor_scores = np.array(impostor_scores)
        
        # Compute metrics
        eer = self.compute_eer(genuine_scores, impostor_scores)
        min_dcf = self.compute_mindcf(genuine_scores, impostor_scores)
        
        # Find threshold at EER
        all_scores = np.concatenate([genuine_scores, impostor_scores])
        labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
        fpr, tpr, thresholds = roc_curve(labels, all_scores)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer_threshold = thresholds[eer_idx]
        
        far_eer, frr_eer = self.compute_far_frr(genuine_scores, impostor_scores, eer_threshold)
        
        # Compute AUC
        roc_auc = auc(fpr, tpr)
        
        # Compute precision-recall AUC
        precision, recall, _ = precision_recall_curve(labels, all_scores)
        pr_auc = average_precision_score(labels, all_scores)
        
        results = {
            'EER': eer,
            'minDCF': min_dcf,
            'FAR_at_EER': far_eer,
            'FRR_at_EER': frr_eer,
            'ROC_AUC': roc_auc,
            'PR_AUC': pr_auc,
            'EER_Threshold': eer_threshold,
            'Genuine_Mean': np.mean(genuine_scores),
            'Impostor_Mean': np.mean(impostor_scores),
            'Genuine_Std': np.std(genuine_scores),
            'Impostor_Std': np.std(impostor_scores)
        }
        
        logger.info(f"{modality} evaluation completed:")
        logger.info(f"  EER: {eer:.4f}")
        logger.info(f"  minDCF: {min_dcf:.4f}")
        logger.info(f"  ROC AUC: {roc_auc:.4f}")
        
        return results
        
    def generate_report(
        self, 
        results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ) -> str:
        """Generate comprehensive evaluation report.
        
        Args:
            results: Evaluation results for each modality
            save_path: Path to save the report
            
        Returns:
            Formatted report string
        """
        report = "Biometric Verification System Evaluation Report\n"
        report += "=" * 50 + "\n\n"
        
        # Summary table
        report += "Summary Metrics:\n"
        report += "-" * 20 + "\n"
        report += f"{'Modality':<12} {'EER':<8} {'minDCF':<8} {'ROC AUC':<8} {'PR AUC':<8}\n"
        report += "-" * 50 + "\n"
        
        for modality, metrics in results.items():
            report += f"{modality:<12} {metrics['EER']:<8.4f} {metrics['minDCF']:<8.4f} "
            report += f"{metrics['ROC_AUC']:<8.4f} {metrics['PR_AUC']:<8.4f}\n"
        
        report += "\nDetailed Results:\n"
        report += "-" * 20 + "\n"
        
        for modality, metrics in results.items():
            report += f"\n{modality.upper()}:\n"
            report += f"  Equal Error Rate (EER): {metrics['EER']:.4f}\n"
            report += f"  Minimum DCF: {metrics['minDCF']:.4f}\n"
            report += f"  FAR at EER: {metrics['FAR_at_EER']:.4f}\n"
            report += f"  FRR at EER: {metrics['FRR_at_EER']:.4f}\n"
            report += f"  ROC AUC: {metrics['ROC_AUC']:.4f}\n"
            report += f"  PR AUC: {metrics['PR_AUC']:.4f}\n"
            report += f"  EER Threshold: {metrics['EER_Threshold']:.4f}\n"
            report += f"  Genuine Score Mean: {metrics['Genuine_Mean']:.4f} ± {metrics['Genuine_Std']:.4f}\n"
            report += f"  Impostor Score Mean: {metrics['Impostor_Mean']:.4f} ± {metrics['Impostor_Std']:.4f}\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
                
        return report


def create_leaderboard(results: Dict[str, Dict[str, float]]) -> str:
    """Create a leaderboard-style comparison of modalities.
    
    Args:
        results: Evaluation results for each modality
        
    Returns:
        Formatted leaderboard string
    """
    leaderboard = "Biometric Verification Leaderboard\n"
    leaderboard += "=" * 40 + "\n\n"
    
    # Sort by EER (lower is better)
    sorted_modalities = sorted(results.items(), key=lambda x: x[1]['EER'])
    
    leaderboard += "Ranking by EER (Equal Error Rate):\n"
    leaderboard += "-" * 30 + "\n"
    
    for rank, (modality, metrics) in enumerate(sorted_modalities, 1):
        leaderboard += f"{rank}. {modality.upper():<12} EER: {metrics['EER']:.4f}\n"
    
    leaderboard += "\nRanking by ROC AUC (higher is better):\n"
    leaderboard += "-" * 35 + "\n"
    
    sorted_by_auc = sorted(results.items(), key=lambda x: x[1]['ROC_AUC'], reverse=True)
    for rank, (modality, metrics) in enumerate(sorted_by_auc, 1):
        leaderboard += f"{rank}. {modality.upper():<12} AUC: {metrics['ROC_AUC']:.4f}\n"
    
    return leaderboard
