"""Module 8: Validation and Evaluation

This module performs comprehensive validation of clustering results
including stability analysis and clinical coherence assessment.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)
from sklearn.cluster import KMeans
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ClusterValidator:
    """Validate clustering results and assess stability."""
    
    def __init__(self, config: Dict):
        """Initialize validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.validation_config = config['validation']
    
    def calculate_internal_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate internal validation metrics.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            
        Returns:
            Dictionary of validation metrics
        """
        logger.info("Calculating internal validation metrics")
        
        # Filter out noise points
        valid_mask = labels != -1
        X_valid = X[valid_mask]
        labels_valid = labels[valid_mask]
        
        n_clusters = len(set(labels_valid))
        
        if n_clusters < 2:
            logger.warning("Less than 2 clusters found, cannot calculate metrics")
            return {}
        
        metrics = {
            'silhouette_score': silhouette_score(X_valid, labels_valid),
            'davies_bouldin_score': davies_bouldin_score(X_valid, labels_valid),
            'calinski_harabasz_score': calinski_harabasz_score(X_valid, labels_valid),
            'n_clusters': n_clusters,
            'n_samples': len(labels_valid),
            'noise_ratio': (len(labels) - len(labels_valid)) / len(labels)
        }
        
        logger.info("Internal metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def bootstrap_stability(self, X: np.ndarray, n_clusters: int, 
                          n_iterations: int = 100) -> Dict[str, float]:
        """Perform bootstrap stability analysis.
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            n_iterations: Number of bootstrap iterations
            
        Returns:
            Dictionary with stability metrics
        """
        logger.info(f"Performing bootstrap stability analysis ({n_iterations} iterations)")
        
        # Original clustering
        original_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        original_labels = original_model.fit_predict(X)
        
        ari_scores = []
        silhouette_scores = []
        
        for i in tqdm(range(n_iterations), desc="Bootstrap iterations"):
            # Bootstrap sample
            n_samples = X.shape[0]
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[indices]
            
            # Cluster bootstrap sample
            bootstrap_model = KMeans(n_clusters=n_clusters, 
                                   random_state=i, 
                                   n_init=10)
            bootstrap_labels = bootstrap_model.fit_predict(X_bootstrap)
            
            # Calculate metrics
            try:
                sil = silhouette_score(X_bootstrap, bootstrap_labels)
                silhouette_scores.append(sil)
            except:
                pass
            
            # Calculate ARI for overlapping samples
            try:
                # Re-predict original samples with bootstrap model
                bootstrap_pred = bootstrap_model.predict(X)
                ari = adjusted_rand_score(original_labels, bootstrap_pred)
                ari_scores.append(ari)
            except:
                pass
        
        stability_metrics = {
            'mean_ari': np.mean(ari_scores),
            'std_ari': np.std(ari_scores),
            'mean_silhouette': np.mean(silhouette_scores),
            'std_silhouette': np.std(silhouette_scores),
            'n_iterations': n_iterations
        }
        
        logger.info("Stability metrics:")
        logger.info(f"  Mean ARI: {stability_metrics['mean_ari']:.4f} ± {stability_metrics['std_ari']:.4f}")
        logger.info(f"  Mean Silhouette: {stability_metrics['mean_silhouette']:.4f} ± {stability_metrics['std_silhouette']:.4f}")
        
        return stability_metrics
    
    def assess_cluster_separation(self, X: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
        """Assess separation between clusters.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            
        Returns:
            DataFrame with pairwise cluster distances
        """
        logger.info("Assessing cluster separation")
        
        unique_labels = sorted(set(labels) - {-1})
        n_clusters = len(unique_labels)
        
        # Calculate cluster centroids
        centroids = {}
        for label in unique_labels:
            cluster_points = X[labels == label]
            centroids[label] = cluster_points.mean(axis=0)
        
        # Calculate pairwise distances
        distances = np.zeros((n_clusters, n_clusters))
        
        for i, label_i in enumerate(unique_labels):
            for j, label_j in enumerate(unique_labels):
                if i != j:
                    dist = np.linalg.norm(centroids[label_i] - centroids[label_j])
                    distances[i, j] = dist
        
        # Create DataFrame
        distance_df = pd.DataFrame(
            distances,
            index=[f"Cluster {l}" for l in unique_labels],
            columns=[f"Cluster {l}" for l in unique_labels]
        )
        
        logger.info(f"Average inter-cluster distance: {distances[distances > 0].mean():.4f}")
        
        return distance_df
    
    def assess_clinical_coherence(self, psych_df: pd.DataFrame, 
                                 labels: np.ndarray) -> Dict[str, float]:
        """Assess clinical coherence of clusters.
        
        Args:
            psych_df: Psychiatric patients DataFrame
            labels: Cluster labels
            
        Returns:
            Dictionary with coherence metrics
        """
        logger.info("Assessing clinical coherence")
        
        psych_df['cluster'] = labels
        
        coherence_metrics = {}
        
        # Within-cluster diagnosis diversity
        diagnosis_diversity = []
        for cluster_id in set(labels) - {-1}:
            cluster_data = psych_df[psych_df['cluster'] == cluster_id]
            avg_diagnoses = cluster_data['num_psych_diagnoses'].mean()
            diagnosis_diversity.append(avg_diagnoses)
        
        coherence_metrics['avg_diagnosis_diversity'] = np.mean(diagnosis_diversity)
        
        # Medication coherence (patients in same cluster have similar medications)
        med_coherence = []
        for cluster_id in set(labels) - {-1}:
            cluster_data = psych_df[psych_df['cluster'] == cluster_id]
            
            # Calculate medication class variance
            med_cols = [col for col in cluster_data.columns if col.startswith('med_')]
            if med_cols:
                med_variance = cluster_data[med_cols].var().mean()
                med_coherence.append(1 / (1 + med_variance))  # Lower variance = higher coherence
        
        coherence_metrics['medication_coherence'] = np.mean(med_coherence) if med_coherence else 0
        
        # Keyword alignment coherence
        keyword_cols = [col for col in psych_df.columns if col.startswith('keyword_count_')]
        keyword_coherence = []
        for cluster_id in set(labels) - {-1}:
            cluster_data = psych_df[psych_df['cluster'] == cluster_id]
            if keyword_cols:
                keyword_variance = cluster_data[keyword_cols].var().mean()
                keyword_coherence.append(1 / (1 + keyword_variance))
        
        coherence_metrics['keyword_coherence'] = np.mean(keyword_coherence) if keyword_coherence else 0
        
        logger.info("Clinical coherence metrics:")
        for metric, value in coherence_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return coherence_metrics
    
    def compare_with_theoretical_clusters(self, psych_df: pd.DataFrame, 
                                         labels: np.ndarray) -> Dict[str, float]:
        """Compare discovered clusters with theoretical framework.
        
        Args:
            psych_df: Psychiatric patients DataFrame
            labels: Cluster labels
            
        Returns:
            Dictionary with comparison metrics
        """
        logger.info("Comparing with theoretical framework")
        
        psych_df['cluster'] = labels
        
        # For each cluster, find dominant theoretical category
        cluster_framework = self.config['cluster_framework']
        framework_flags = [f'flag_{name}' for name in cluster_framework.keys()]
        
        alignment_scores = []
        
        for cluster_id in set(labels) - {-1}:
            cluster_data = psych_df[psych_df['cluster'] == cluster_id]
            
            # Find which theoretical category is most prevalent
            flag_counts = {}
            for flag in framework_flags:
                if flag in cluster_data.columns:
                    flag_counts[flag] = cluster_data[flag].sum()
            
            if flag_counts:
                max_count = max(flag_counts.values())
                alignment_score = max_count / len(cluster_data)
                alignment_scores.append(alignment_score)
        
        comparison_metrics = {
            'mean_framework_alignment': np.mean(alignment_scores) if alignment_scores else 0,
            'std_framework_alignment': np.std(alignment_scores) if alignment_scores else 0
        }
        
        logger.info("Framework comparison:")
        logger.info(f"  Mean alignment: {comparison_metrics['mean_framework_alignment']:.4f}")
        
        return comparison_metrics
    
    def generate_validation_report(self, X: np.ndarray, labels: np.ndarray,
                                  psych_df: pd.DataFrame) -> Dict:
        """Generate comprehensive validation report.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            psych_df: Psychiatric patients DataFrame
            
        Returns:
            Dictionary with all validation results
        """
        logger.info("Generating comprehensive validation report")
        
        n_clusters = len(set(labels) - {-1})
        
        report = {
            'internal_metrics': self.calculate_internal_metrics(X, labels),
            'cluster_separation': self.assess_cluster_separation(X, labels),
            'clinical_coherence': self.assess_clinical_coherence(psych_df, labels),
        }
        
        # Add stability analysis if configured
        n_bootstrap = self.validation_config.get('bootstrap_iterations', 0)
        if n_bootstrap > 0:
            report['stability'] = self.bootstrap_stability(X, n_clusters, n_bootstrap)
        
        # Save results
        output_dir = f"{self.config['data']['output_dir']}/reports"
        
        # Save metrics as CSV
        metrics_df = pd.DataFrame([report['internal_metrics']])
        metrics_df.to_csv(f"{output_dir}/validation_metrics.csv", index=False)
        
        # Save separation matrix
        report['cluster_separation'].to_csv(f"{output_dir}/cluster_separation.csv")
        
        logger.info(f"Validation report saved to {output_dir}")
        
        return report
    
    def process(self, X: np.ndarray, labels: np.ndarray, 
               psych_df: pd.DataFrame) -> Dict:
        """Run full validation pipeline.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            psych_df: Psychiatric patients DataFrame
            
        Returns:
            Validation report dictionary
        """
        logger.info("Starting cluster validation")
        
        report = self.generate_validation_report(X, labels, psych_df)
        
        logger.info("Cluster validation complete")
        
        return report
