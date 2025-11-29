"""Module 7: Visualization and Interpretation

This module creates comprehensive visualizations for cluster analysis.
"""

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ClusterVisualizer:
    """Create visualizations for psychiatric clusters."""
    
    def __init__(self, config: Dict):
        """Initialize visualizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.viz_config = config['visualization']
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = self.viz_config['figure_size']
        plt.rcParams['figure.dpi'] = self.viz_config['dpi']
        
        self.output_dir = f"{config['data']['output_dir']}/visualizations"
    
    def plot_2d_clusters(self, embeddings: np.ndarray, labels: np.ndarray, 
                        method: str = 'tsne') -> None:
        """Create 2D visualization of clusters.
        
        Args:
            embeddings: High-dimensional embeddings
            labels: Cluster labels
            method: Dimensionality reduction method ('tsne' or 'umap')
        """
        logger.info(f"Creating 2D cluster visualization using {method.upper()}")
        
        # Reduce to 2D
        if method == 'tsne':
            from sklearn.manifold import TSNE
            tsne_config = self.viz_config['tsne']
            reducer = TSNE(
                n_components=tsne_config['n_components'],
                perplexity=tsne_config['perplexity'],
                random_state=tsne_config['random_state']
            )
            coords_2d = reducer.fit_transform(embeddings)
        
        elif method == 'umap':
            try:
                import umap
                umap_config = self.viz_config['umap_viz']
                reducer = umap.UMAP(
                    n_neighbors=umap_config['n_neighbors'],
                    min_dist=umap_config['min_dist'],
                    n_components=umap_config['n_components'],
                    random_state=umap_config['random_state']
                )
                coords_2d = reducer.fit_transform(embeddings)
            except ImportError:
                logger.warning("UMAP not available, using t-SNE")
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42)
                coords_2d = reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create scatter plot
        plt.figure(figsize=(12, 10))
        
        # Plot each cluster
        unique_labels = sorted(set(labels))
        colors = sns.color_palette(self.viz_config['color_palette'], len(unique_labels))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            
            if label == -1:
                # Plot noise points differently
                plt.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                          c='gray', alpha=0.3, s=20, label='Noise')
            else:
                plt.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                          c=[colors[i]], alpha=0.6, s=50, 
                          label=f'Cluster {label} (n={mask.sum()})')
        
        plt.xlabel(f'{method.upper()} Component 1', fontsize=12)
        plt.ylabel(f'{method.upper()} Component 2', fontsize=12)
        plt.title(f'Psychiatric Disorder Clusters - {method.upper()} Visualization', 
                 fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        output_path = f"{self.output_dir}/clusters_{method}_2d.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"2D visualization saved to {output_path}")
        plt.close()
    
    def plot_cluster_sizes(self, labels: np.ndarray) -> None:
        """Plot cluster size distribution.
        
        Args:
            labels: Cluster labels
        """
        logger.info("Creating cluster size plot")
        
        # Count cluster sizes
        unique, counts = np.unique(labels[labels != -1], return_counts=True)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(unique, counts, color=sns.color_palette(self.viz_config['color_palette'], len(unique)))
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}\n({height/len(labels)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Cluster', fontsize=12)
        plt.ylabel('Number of Patients', fontsize=12)
        plt.title('Cluster Size Distribution', fontsize=14, fontweight='bold')
        plt.xticks(unique)
        plt.grid(axis='y', alpha=0.3)
        
        output_path = f"{self.output_dir}/cluster_sizes.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Cluster sizes plot saved to {output_path}")
        plt.close()
    
    def plot_framework_heatmap(self, framework_comparison: pd.DataFrame) -> None:
        """Plot heatmap of cluster alignment with theoretical framework.
        
        Args:
            framework_comparison: Framework comparison DataFrame
        """
        logger.info("Creating framework comparison heatmap")
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(framework_comparison, annot=True, fmt='.1f', cmap='YlOrRd',
                   cbar_kws={'label': 'Overlap Percentage (%)'},
                   linewidths=0.5, linecolor='gray')
        
        plt.xlabel('Theoretical Framework Category', fontsize=12)
        plt.ylabel('Discovered Cluster', fontsize=12)
        plt.title('Cluster Alignment with Theoretical Framework', 
                 fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_path = f"{self.output_dir}/framework_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Framework heatmap saved to {output_path}")
        plt.close()
    
    def plot_medication_profiles(self, medication_df: pd.DataFrame, top_n: int = 10) -> None:
        """Plot top medications for each cluster.
        
        Args:
            medication_df: Medication analysis DataFrame
            top_n: Number of top medications to plot
        """
        logger.info("Creating medication profile plots")
        
        clusters = sorted(medication_df['cluster'].unique())
        n_clusters = len(clusters)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, cluster_id in enumerate(clusters[:6]):  # Max 6 subplots
            cluster_meds = medication_df[medication_df['cluster'] == cluster_id].head(top_n)
            
            ax = axes[idx]
            bars = ax.barh(range(len(cluster_meds)), cluster_meds['percentage'])
            ax.set_yticks(range(len(cluster_meds)))
            ax.set_yticklabels(cluster_meds['medication'], fontsize=9)
            ax.set_xlabel('Percentage of Patients', fontsize=10)
            ax.set_title(f'Cluster {cluster_id} - Top Medications', 
                        fontsize=11, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Color bars
            for bar in bars:
                bar.set_color(sns.color_palette(self.viz_config['color_palette'], n_clusters)[idx])
        
        # Hide unused subplots
        for idx in range(len(clusters), 6):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        output_path = f"{self.output_dir}/medication_profiles.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Medication profiles saved to {output_path}")
        plt.close()
    
    def plot_diagnosis_distribution(self, diagnosis_df: pd.DataFrame, top_n: int = 8) -> None:
        """Plot top diagnoses for each cluster.
        
        Args:
            diagnosis_df: Diagnosis analysis DataFrame
            top_n: Number of top diagnoses to plot
        """
        logger.info("Creating diagnosis distribution plots")
        
        clusters = sorted(diagnosis_df['cluster'].unique())
        n_clusters = len(clusters)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, cluster_id in enumerate(clusters[:6]):
            cluster_diags = diagnosis_df[diagnosis_df['cluster'] == cluster_id].head(top_n)
            
            ax = axes[idx]
            bars = ax.barh(range(len(cluster_diags)), cluster_diags['percentage'])
            ax.set_yticks(range(len(cluster_diags)))
            ax.set_yticklabels(cluster_diags['icd9_code'], fontsize=9)
            ax.set_xlabel('Percentage of Patients', fontsize=10)
            ax.set_title(f'Cluster {cluster_id} - Top ICD-9 Codes', 
                        fontsize=11, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Color bars
            for bar in bars:
                bar.set_color(sns.color_palette(self.viz_config['color_palette'], n_clusters)[idx])
        
        # Hide unused subplots
        for idx in range(len(clusters), 6):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        output_path = f"{self.output_dir}/diagnosis_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Diagnosis distribution saved to {output_path}")
        plt.close()
    
    def plot_cluster_statistics(self, statistics_df: pd.DataFrame) -> None:
        """Plot key statistics comparison across clusters.
        
        Args:
            statistics_df: Cluster statistics DataFrame
        """
        logger.info("Creating cluster statistics plots")
        
        # Select key metrics
        metrics = [
            'avg_num_diagnoses',
            'avg_num_medications',
            'polypharmacy_rate',
            'comorbidity_rate'
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            bars = ax.bar(statistics_df['cluster'], statistics_df[metric],
                         color=sns.color_palette(self.viz_config['color_palette'], 
                                                len(statistics_df)))
            
            ax.set_xlabel('Cluster', fontsize=11)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
            ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        output_path = f"{self.output_dir}/cluster_statistics.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Cluster statistics saved to {output_path}")
        plt.close()
    
    def create_word_clouds(self, cluster_keywords: Dict[int, list]) -> None:
        """Create word clouds for each cluster.
        
        Args:
            cluster_keywords: Dictionary mapping cluster_id to keywords
        """
        logger.info("Creating word clouds")
        
        try:
            from wordcloud import WordCloud
            
            n_clusters = len(cluster_keywords)
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for idx, (cluster_id, keywords) in enumerate(sorted(cluster_keywords.items())[:6]):
                # Create word frequency dict
                word_freq = {word: len(keywords) - i for i, word in enumerate(keywords[:30])}
                
                # Generate word cloud
                wordcloud = WordCloud(
                    width=400, 
                    height=300,
                    background_color='white',
                    colormap='viridis'
                ).generate_from_frequencies(word_freq)
                
                ax = axes[idx]
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_title(f'Cluster {cluster_id} - Key Terms', 
                           fontsize=12, fontweight='bold')
                ax.axis('off')
            
            # Hide unused subplots
            for idx in range(len(cluster_keywords), 6):
                axes[idx].axis('off')
            
            plt.tight_layout()
            
            output_path = f"{self.output_dir}/cluster_wordclouds.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Word clouds saved to {output_path}")
            plt.close()
            
        except ImportError:
            logger.warning("wordcloud library not available, skipping word cloud generation")
    
    def process(self, embeddings: np.ndarray, labels: np.ndarray,
               characterization_report: Dict) -> None:
        """Generate all visualizations.
        
        Args:
            embeddings: Feature embeddings
            labels: Cluster labels
            characterization_report: Characterization report dictionary
        """
        logger.info("Starting visualization generation")
        
        # 2D cluster visualizations
        self.plot_2d_clusters(embeddings, labels, method='tsne')
        self.plot_2d_clusters(embeddings, labels, method='umap')
        
        # Cluster sizes
        self.plot_cluster_sizes(labels)
        
        # Framework comparison
        self.plot_framework_heatmap(characterization_report['framework_comparison'])
        
        # Medication and diagnosis profiles
        self.plot_medication_profiles(characterization_report['medications'])
        self.plot_diagnosis_distribution(characterization_report['diagnoses'])
        
        # Statistics
        self.plot_cluster_statistics(characterization_report['statistics'])
        
        # Word clouds
        self.create_word_clouds(characterization_report['keywords'])
        
        logger.info(f"All visualizations saved to {self.output_dir}")
