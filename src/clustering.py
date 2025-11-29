"""Module 5: Unsupervised Clustering

This module implements multiple clustering algorithms and
selects the optimal clustering solution.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class PsychiatricClusterer:
    """Perform clustering on psychiatric patient embeddings."""
    
    def __init__(self, config: Dict):
        """Initialize clusterer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.cluster_config = config['clustering']
        self.validation_config = config['validation']
        
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_labels = None
        self.best_algorithm = None
        self.best_score = -np.inf
    
    def prepare_features(self, embeddings: np.ndarray) -> np.ndarray:
        """Prepare and scale features for clustering.
        
        Args:
            embeddings: Raw embeddings
            
        Returns:
            Scaled embeddings
        """
        logger.info("Scaling features for clustering")
        scaled_embeddings = self.scaler.fit_transform(embeddings)
        return scaled_embeddings
    
    def run_kmeans(self, X: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, object]:
        """Run K-Means clustering.
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            
        Returns:
            Tuple of (cluster labels, model)
        """
        logger.info(f"Running K-Means with k={n_clusters}")
        
        kmeans_config = self.cluster_config['kmeans']
        
        model = KMeans(
            n_clusters=n_clusters,
            n_init=kmeans_config['n_init'],
            max_iter=kmeans_config['max_iter'],
            random_state=kmeans_config['random_state']
        )
        
        labels = model.fit_predict(X)
        
        return labels, model
    
    def run_hierarchical(self, X: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, object]:
        """Run Hierarchical clustering.
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            
        Returns:
            Tuple of (cluster labels, model)
        """
        logger.info(f"Running Hierarchical clustering with n={n_clusters}")
        
        hier_config = self.cluster_config['hierarchical']
        '''
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=hier_config['linkage'],
            affinity=hier_config['affinity']
        )
        '''
        model = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')

        labels = model.fit_predict(X)
        
        return labels, model
    
    def run_dbscan(self, X: np.ndarray) -> Tuple[np.ndarray, object]:
        """Run DBSCAN clustering.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (cluster labels, model)
        """
        logger.info("Running DBSCAN")
        
        dbscan_config = self.cluster_config['dbscan']
        
        model = DBSCAN(
            eps=dbscan_config['eps'],
            min_samples=dbscan_config['min_samples'],
            metric=dbscan_config['metric']
        )
        
        labels = model.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        logger.info(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points")
        
        return labels, model
    
    def run_gmm(self, X: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, object]:
        """Run Gaussian Mixture Model clustering.
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            
        Returns:
            Tuple of (cluster labels, model)
        """
        logger.info(f"Running GMM with n={n_clusters}")
        
        gmm_config = self.cluster_config['gmm']
        '''
        model = GaussianMixture(
            n_components=n_clusters,
            covariance_type=gmm_config['covariance_type'],
            n_init=gmm_config['n_init'],
            random_state=gmm_config['random_state']
        )
        '''
        model = GaussianMixture(
            n_components=n_clusters,
            covariance_type=gmm_config.get('covariance_type', 'full'),
            n_init=gmm_config.get('n_init', 5),
            random_state=gmm_config.get('random_state', 42),
            reg_covar=1e-3  # IMPORTANT FIX
        )

        model.fit(X)
        labels = model.predict(X)
        
        return labels, model
    
    def calculate_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering validation metrics.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            
        Returns:
            Dictionary of metrics
        """
        # Filter out noise points (label -1) if present
        valid_mask = labels != -1
        X_valid = X[valid_mask]
        labels_valid = labels[valid_mask]
        
        # Need at least 2 clusters for metrics
        n_clusters = len(set(labels_valid))
        if n_clusters < 2:
            return {
                'silhouette_score': -1.0,
                'davies_bouldin_score': np.inf,
                'calinski_harabasz_score': 0.0,
                'n_clusters': n_clusters
            }
        
        metrics = {
            'silhouette_score': silhouette_score(X_valid, labels_valid),
            'davies_bouldin_score': davies_bouldin_score(X_valid, labels_valid),
            'calinski_harabasz_score': calinski_harabasz_score(X_valid, labels_valid),
            'n_clusters': n_clusters
        }
        
        return metrics
    
    def elbow_method(self, X: np.ndarray) -> None:
        """Run elbow method to determine optimal k.
        
        Args:
            X: Feature matrix
        """
        logger.info("Running elbow method")
        
        k_range = self.validation_config['test_k_range']
        inertias = []
        silhouettes = []
        
        for k in k_range:
            labels, model = self.run_kmeans(X, k)
            inertias.append(model.inertia_)
            
            if k > 1:
                sil_score = silhouette_score(X, labels)
                silhouettes.append(sil_score)
            else:
                silhouettes.append(0)
        
        # Plot elbow curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(k_range, inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method - Inertia')
        ax1.grid(True)
        
        ax2.plot(k_range, silhouettes, 'ro-')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Elbow Method - Silhouette Score')
        ax2.grid(True)
        
        output_path = f"{self.config['data']['output_dir']}/visualizations/elbow_method.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Elbow method plot saved to {output_path}")
        plt.close()
    
    def compare_algorithms(self, X: np.ndarray) -> pd.DataFrame:
        """Compare different clustering algorithms.
        
        Args:
            X: Feature matrix
            
        Returns:
            DataFrame with comparison results
        """
        logger.info("Comparing clustering algorithms")
        
        n_clusters = self.cluster_config['n_clusters']
        algorithms = self.cluster_config['algorithms']
        
        results = []
        
        for algo in algorithms:
            logger.info(f"Testing {algo}")
            
            try:
                if algo == 'kmeans':
                    labels, model = self.run_kmeans(X, n_clusters)
                elif algo == 'hierarchical':
                    labels, model = self.run_hierarchical(X, n_clusters)
                elif algo == 'dbscan':
                    labels, model = self.run_dbscan(X)
                elif algo == 'gmm':
                    labels, model = self.run_gmm(X, n_clusters)
                else:
                    continue
                
                metrics = self.calculate_metrics(X, labels)
                
                result = {
                    'algorithm': algo,
                    **metrics
                }
                results.append(result)
                
                logger.info(f"  Silhouette: {metrics['silhouette_score']:.3f}")
                logger.info(f"  Davies-Bouldin: {metrics['davies_bouldin_score']:.3f}")
                logger.info(f"  Calinski-Harabasz: {metrics['calinski_harabasz_score']:.1f}")
                
                # Track best model based on silhouette score
                if metrics['silhouette_score'] > self.best_score:
                    self.best_score = metrics['silhouette_score']
                    self.best_model = model
                    self.best_labels = labels
                    self.best_algorithm = algo
                
            except Exception as e:
                logger.error(f"Error running {algo}: {e}")
        
        results_df = pd.DataFrame(results)
        
        logger.info(f"\nBest algorithm: {self.best_algorithm} (Silhouette: {self.best_score:.3f})")
        
        return results_df
    
    def process(self, embeddings: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
        """Run full clustering pipeline.
        
        Args:
            embeddings: Feature embeddings
            
        Returns:
            Tuple of (cluster labels, comparison results)
        """
        logger.info("Starting clustering analysis")
        
        # Prepare features
        X = self.prepare_features(embeddings)
        
        # Run elbow method
        self.elbow_method(X)
        
        # Compare algorithms
        comparison_df = self.compare_algorithms(X)
        
        # Save comparison results
        output_path = f"{self.config['data']['output_dir']}/clusters/algorithm_comparison.csv"
        comparison_df.to_csv(output_path, index=False)
        logger.info(f"Algorithm comparison saved to {output_path}")
        
        logger.info(f"Clustering complete. Using {self.best_algorithm} with {len(set(self.best_labels))} clusters")
        
        return self.best_labels, comparison_df
