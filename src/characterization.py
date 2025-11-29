"""Module 6: Cluster Characterization

This module profiles each cluster with diagnostic codes,
medications, keywords, and statistical characteristics.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List
from collections import Counter
from scipy import stats
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ClusterCharacterizer:
    """Characterize and profile psychiatric clusters."""
    
    def __init__(self, config: Dict):
        """Initialize cluster characterizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.cluster_framework = config['cluster_framework']
    
    def analyze_diagnoses(self, psych_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        """Analyze top diagnoses for each cluster.
        
        Args:
            psych_df: Psychiatric patients DataFrame
            labels: Cluster labels
            
        Returns:
            DataFrame with top diagnoses per cluster
        """
        logger.info("Analyzing diagnosis patterns")
        
        psych_df['cluster'] = labels
        
        diagnosis_results = []
        
        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:  # Skip noise
                continue
            
            cluster_data = psych_df[psych_df['cluster'] == cluster_id]
            
            # Collect all diagnoses for this cluster
            all_diagnoses = []
            for diag_list in cluster_data['all_psych_diagnoses']:
                if isinstance(diag_list, list):
                    all_diagnoses.extend(diag_list)
            
            # Count diagnoses
            diag_counts = Counter(all_diagnoses)
            
            # Get top 10
            top_diagnoses = diag_counts.most_common(10)
            
            for rank, (diag, count) in enumerate(top_diagnoses, 1):
                diagnosis_results.append({
                    'cluster': cluster_id,
                    'rank': rank,
                    'icd9_code': diag,
                    'count': count,
                    'percentage': count / len(cluster_data) * 100
                })
        
        diagnosis_df = pd.DataFrame(diagnosis_results)
        
        logger.info(f"Analyzed diagnoses for {len(set(labels) - {-1})} clusters")
        
        return diagnosis_df
    
    def analyze_medications(self, psych_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        """Analyze top medications for each cluster.
        
        Args:
            psych_df: Psychiatric patients DataFrame
            labels: Cluster labels
            
        Returns:
            DataFrame with top medications per cluster
        """
        logger.info("Analyzing medication patterns")
        
        medication_results = []
        
        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:
                continue
            
            cluster_data = psych_df[psych_df['cluster'] == cluster_id]
            
            # Collect all medications
            all_meds = []
            for med_list in cluster_data['medications']:
                if isinstance(med_list, list):
                    all_meds.extend([m.lower() for m in med_list])
            
            # Count medications
            med_counts = Counter(all_meds)
            
            # Get top 15
            top_meds = med_counts.most_common(15)
            
            for rank, (med, count) in enumerate(top_meds, 1):
                medication_results.append({
                    'cluster': cluster_id,
                    'rank': rank,
                    'medication': med,
                    'count': count,
                    'percentage': count / len(cluster_data) * 100
                })
        
        medication_df = pd.DataFrame(medication_results)
        
        logger.info(f"Analyzed medications for {len(set(labels) - {-1})} clusters")
        
        return medication_df
    
    def extract_cluster_keywords(self, psych_df: pd.DataFrame, labels: np.ndarray, 
                                 top_n: int = 50) -> Dict[int, List[str]]:
        """Extract characteristic keywords for each cluster using TF-IDF.
        
        Args:
            psych_df: Psychiatric patients DataFrame
            labels: Cluster labels
            top_n: Number of top keywords to extract
            
        Returns:
            Dictionary mapping cluster_id to list of keywords
        """
        logger.info("Extracting characteristic keywords")
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        cluster_keywords = {}
        
        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:
                continue
            
            # Get texts for this cluster
            cluster_texts = psych_df[psych_df['cluster'] == cluster_id]['processed_text'].tolist()
            '''
            other_texts = psych_df[psych_df['cluster'] != cluster_id]['processed_text'].sample(
                min(len(cluster_texts), 500), random_state=42
            ).tolist()
            '''
            other_pool = psych_df[psych_df['cluster'] != cluster_id]['processed_text'].dropna()

            sample_size = min(len(cluster_texts), 500, len(other_pool))

            if sample_size == 0:
                other_texts = []
            else:
                other_texts = other_pool.sample(sample_size, replace=False, random_state=42).tolist()

            # Combine texts
            all_texts = cluster_texts + other_texts
            labels_binary = [1] * len(cluster_texts) + [0] * len(other_texts)
            
            # TF-IDF vectorization
            try:
                vectorizer = TfidfVectorizer(max_features=500, min_df=2, max_df=0.8)
                tfidf_matrix = vectorizer.fit_transform(all_texts)
                feature_names = vectorizer.get_feature_names_out()
                
                # Calculate mean TF-IDF for cluster vs others
                cluster_mask = np.array(labels_binary) == 1
                cluster_tfidf = tfidf_matrix[cluster_mask].mean(axis=0).A1
                other_tfidf = tfidf_matrix[~cluster_mask].mean(axis=0).A1
                
                # Calculate difference
                tfidf_diff = cluster_tfidf - other_tfidf
                
                # Get top keywords
                top_indices = tfidf_diff.argsort()[-top_n:][::-1]
                top_keywords = [feature_names[i] for i in top_indices]
                
                cluster_keywords[cluster_id] = top_keywords
                
            except Exception as e:
                logger.warning(f"Could not extract keywords for cluster {cluster_id}: {e}")
                cluster_keywords[cluster_id] = []
        
        return cluster_keywords
    
    def calculate_cluster_statistics(self, psych_df: pd.DataFrame, 
                                    labels: np.ndarray) -> pd.DataFrame:
        """Calculate statistical profile for each cluster.
        
        Args:
            psych_df: Psychiatric patients DataFrame
            labels: Cluster labels
            
        Returns:
            DataFrame with cluster statistics
        """
        logger.info("Calculating cluster statistics")
        
        stats_results = []
        
        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:
                continue
            
            cluster_data = psych_df[psych_df['cluster'] == cluster_id]
            
            # Calculate statistics
            stats_dict = {
                'cluster': cluster_id,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(psych_df) * 100,
                
                # Diagnosis statistics
                'avg_num_diagnoses': cluster_data['num_psych_diagnoses'].mean(),
                'avg_num_diagnosis_groups': cluster_data['num_diagnosis_groups'].mean(),
                
                # Medication statistics
                'avg_num_medications': cluster_data['num_medications'].mean(),
                'polypharmacy_rate': cluster_data['polypharmacy'].mean() * 100,
                
                # Comorbidity
                'comorbidity_rate': cluster_data['has_comorbidity'].mean() * 100,
                'avg_cluster_categories': cluster_data['num_cluster_categories'].mean(),
                
                # Text statistics
                'avg_text_length': cluster_data['text_length'].mean(),
                'avg_word_count': cluster_data['word_count'].mean(),
            }
            
            # Add keyword counts
            for cluster_name in self.cluster_framework.keys():
                keyword_col = f'keyword_count_{cluster_name}'
                if keyword_col in cluster_data.columns:
                    stats_dict[f'avg_{keyword_col}'] = cluster_data[keyword_col].mean()
            
            # Add medication class prevalence
            med_cols = [col for col in cluster_data.columns if col.startswith('med_')]
            for med_col in med_cols:
                prevalence = (cluster_data[med_col] > 0).mean() * 100
                stats_dict[f'{med_col}_prevalence'] = prevalence
            
            stats_results.append(stats_dict)
        
        stats_df = pd.DataFrame(stats_results)
        
        logger.info(f"Calculated statistics for {len(stats_df)} clusters")
        
        return stats_df
    
    def compare_with_framework(self, psych_df: pd.DataFrame, 
                              labels: np.ndarray) -> pd.DataFrame:
        """Compare clusters with theoretical framework.
        
        Args:
            psych_df: Psychiatric patients DataFrame
            labels: Cluster labels
            
        Returns:
            DataFrame with framework comparison
        """
        logger.info("Comparing clusters with theoretical framework")
        
        comparison_results = []
        
        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:
                continue
            
            cluster_data = psych_df[psych_df['cluster'] == cluster_id]
            
            # Check overlap with each framework category
            for framework_name, framework_info in self.cluster_framework.items():
                flag_col = f'flag_{framework_name}'
                
                if flag_col in cluster_data.columns:
                    overlap = cluster_data[flag_col].sum()
                    overlap_pct = overlap / len(cluster_data) * 100
                    
                    comparison_results.append({
                        'cluster': cluster_id,
                        'framework_category': framework_info['name'],
                        'overlap_count': overlap,
                        'overlap_percentage': overlap_pct
                    })
        
        comparison_df = pd.DataFrame(comparison_results)
        
        # Pivot for better visualization
        pivot_df = comparison_df.pivot(
            index='cluster', 
            columns='framework_category', 
            values='overlap_percentage'
        )
        
        logger.info("Framework comparison complete")
        
        return pivot_df
    
    def generate_cluster_report(self, psych_df: pd.DataFrame, labels: np.ndarray) -> Dict:
        """Generate comprehensive cluster characterization report.
        
        Args:
            psych_df: Psychiatric patients DataFrame
            labels: Cluster labels
            
        Returns:
            Dictionary with all characterization results
        """
        logger.info("Generating comprehensive cluster report")
        
        psych_df['cluster'] = labels
        
        report = {
            'diagnoses': self.analyze_diagnoses(psych_df, labels),
            'medications': self.analyze_medications(psych_df, labels),
            'keywords': self.extract_cluster_keywords(psych_df, labels),
            'statistics': self.calculate_cluster_statistics(psych_df, labels),
            'framework_comparison': self.compare_with_framework(psych_df, labels)
        }
        
        # Save all results
        output_dir = f"{self.config['data']['output_dir']}/reports"
        
        report['diagnoses'].to_csv(f"{output_dir}/cluster_diagnoses.csv", index=False)
        report['medications'].to_csv(f"{output_dir}/cluster_medications.csv", index=False)
        report['statistics'].to_csv(f"{output_dir}/cluster_statistics.csv", index=False)
        report['framework_comparison'].to_csv(f"{output_dir}/framework_comparison.csv")
        
        # Save keywords as text
        with open(f"{output_dir}/cluster_keywords.txt", 'w') as f:
            for cluster_id, keywords in report['keywords'].items():
                f.write(f"\nCluster {cluster_id}:\n")
                f.write(", ".join(keywords[:30]))
                f.write("\n")
        
        logger.info(f"Cluster report saved to {output_dir}")
        
        return report
    
    def process(self, psych_df: pd.DataFrame, labels: np.ndarray) -> Dict:
        """Run full characterization pipeline.
        
        Args:
            psych_df: Psychiatric patients DataFrame
            labels: Cluster labels
            
        Returns:
            Characterization report dictionary
        """
        logger.info("Starting cluster characterization")
        
        report = self.generate_cluster_report(psych_df, labels)
        
        logger.info("Cluster characterization complete")
        
        return report
