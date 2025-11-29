"""Module 3: Feature Engineering

This module extracts clinical features from notes, medications,
and diagnoses for clustering.
"""

import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, List, Set
from tqdm import tqdm
from collections import Counter

logger = logging.getLogger(__name__)


class ClinicalFeatureEngineer:
    """Extract clinical features for clustering."""
    
    def __init__(self, config: Dict):
        """Initialize feature engineer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.cluster_framework = config['cluster_framework']
        self.text_config = config['text_processing']
        self.features_config = config['features']
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess clinical text.
        
        Args:
            text: Raw clinical text
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        if self.text_config.get('lowercase', True):
            text = text.lower()
        
        # Remove special characters (keep alphanumeric and spaces)
        if self.text_config.get('remove_special_chars', True):
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_keyword_features(self, psych_df: pd.DataFrame) -> pd.DataFrame:
        """Extract keyword-based features for each cluster type.
        
        Args:
            psych_df: Psychiatric patients DataFrame
            
        Returns:
            DataFrame with keyword features
        """
        logger.info("Extracting keyword features from discharge summaries")
        
        # Preprocess all discharge summaries
        psych_df['processed_text'] = psych_df['discharge_summary'].apply(self.preprocess_text)
        
        # Extract keyword counts for each cluster
        for cluster_name, cluster_info in self.cluster_framework.items():
            keywords = cluster_info['keywords']
            feature_name = f'keyword_count_{cluster_name}'
            
            def count_keywords(text: str) -> int:
                """Count keyword occurrences."""
                count = 0
                for keyword in keywords:
                    count += text.count(keyword.lower())
                return count
            
            psych_df[feature_name] = psych_df['processed_text'].apply(count_keywords)
        
        # Calculate total keyword counts
        keyword_cols = [col for col in psych_df.columns if col.startswith('keyword_count_')]
        psych_df['total_keyword_count'] = psych_df[keyword_cols].sum(axis=1)
        
        logger.info(f"Extracted keyword features for {len(self.cluster_framework)} clusters")
        
        return psych_df
    
    def extract_text_statistics(self, psych_df: pd.DataFrame) -> pd.DataFrame:
        """Extract text length and complexity features.
        
        Args:
            psych_df: Psychiatric patients DataFrame
            
        Returns:
            DataFrame with text statistics
        """
        logger.info("Extracting text statistics")
        
        # Text length features
        psych_df['text_length'] = psych_df['discharge_summary'].str.len()
        psych_df['word_count'] = psych_df['processed_text'].str.split().str.len()
        psych_df['avg_word_length'] = psych_df['processed_text'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x else 0
        )
        
        # Sentence count (approximate)
        psych_df['sentence_count'] = psych_df['discharge_summary'].str.count(r'[.!?]')
        
        logger.info("Text statistics extracted")
        
        return psych_df
    
    def categorize_medications(self, medications: List[str]) -> Dict[str, int]:
        """Categorize medications into drug classes.
        
        Args:
            medications: List of medication names
            
        Returns:
            Dictionary of drug class counts
        """
        if not isinstance(medications, list):
            medications = []
        
        # Lowercase medication names
        medications_lower = [med.lower() for med in medications]
        
        # Count medications in each class
        class_counts = {}
        
        for drug_class, drug_list in self.features_config['medication_classes'].items():
            count = 0
            for drug in drug_list:
                for med in medications_lower:
                    if drug.lower() in med:
                        count += 1
                        break
            class_counts[drug_class] = count
        
        return class_counts
    
    def extract_medication_features(self, psych_df: pd.DataFrame) -> pd.DataFrame:
        """Extract medication-based features.
        
        Args:
            psych_df: Psychiatric patients DataFrame
            
        Returns:
            DataFrame with medication features
        """
        logger.info("Extracting medication features")
        
        # Categorize medications for each patient
        med_categories = psych_df['medications'].apply(self.categorize_medications)
        
        # Convert to columns
        med_df = pd.DataFrame(med_categories.tolist())
        med_df.columns = [f'med_{col}' for col in med_df.columns]
        
        # Merge with main dataframe
        psych_df = pd.concat([psych_df, med_df], axis=1)
        
        # Polypharmacy indicator (5+ medications)
        psych_df['polypharmacy'] = (psych_df['num_medications'] >= 5).astype(int)
        
        logger.info(f"Extracted {len(med_df.columns)} medication class features")
        
        return psych_df
    
    def extract_medication_keyword_alignment(self, psych_df: pd.DataFrame) -> pd.DataFrame:
        """Extract alignment between medications and keyword patterns.
        
        Args:
            psych_df: Psychiatric patients DataFrame
            
        Returns:
            DataFrame with alignment features
        """
        logger.info("Extracting medication-keyword alignment")
        
        for cluster_name, cluster_info in self.cluster_framework.items():
            keyword_col = f'keyword_count_{cluster_name}'
            
            # Check if patient has medications from this cluster
            cluster_meds = cluster_info['medications']
            
            def has_cluster_medication(medications: List[str]) -> int:
                """Check if patient has any medication from cluster."""
                if not isinstance(medications, list):
                    return 0
                
                medications_lower = [m.lower() for m in medications]
                for med in cluster_meds:
                    for patient_med in medications_lower:
                        if med.lower() in patient_med:
                            return 1
                return 0
            
            med_flag = f'has_med_{cluster_name}'
            psych_df[med_flag] = psych_df['medications'].apply(has_cluster_medication)
            
            # Alignment score (both keywords and medications present)
            alignment_col = f'alignment_{cluster_name}'
            psych_df[alignment_col] = (
                (psych_df[keyword_col] > 0) & (psych_df[med_flag] > 0)
            ).astype(int)
        
        return psych_df
    
    def extract_diagnosis_features(self, psych_df: pd.DataFrame) -> pd.DataFrame:
        """Extract diagnosis-based features.
        
        Args:
            psych_df: Psychiatric patients DataFrame
            
        Returns:
            DataFrame with diagnosis features
        """
        logger.info("Extracting diagnosis features")
        
        # Group diagnoses by ICD-9 chapter (first 3 digits)
        def get_diagnosis_groups(diagnoses: List[str]) -> List[str]:
            """Get diagnosis groups."""
            if not isinstance(diagnoses, list):
                return []
            
            groups = set()
            for diag in diagnoses:
                try:
                    # Extract first 3 digits
                    diag_str = str(diag).split('.')[0]
                    if diag_str.isdigit() and len(diag_str) >= 3:
                        groups.add(diag_str[:3])
                except:
                    continue
            
            return list(groups)
        
        psych_df['diagnosis_groups'] = psych_df['all_psych_diagnoses'].apply(get_diagnosis_groups)
        psych_df['num_diagnosis_groups'] = psych_df['diagnosis_groups'].apply(len)
        
        return psych_df
    
    def create_feature_matrix(self, psych_df: pd.DataFrame) -> pd.DataFrame:
        """Create final feature matrix for clustering.
        
        Args:
            psych_df: Psychiatric patients DataFrame
            
        Returns:
            DataFrame with all features
        """
        logger.info("Creating comprehensive feature matrix")
        
        # Select feature columns
        feature_cols = [
            # Keyword features
            *[col for col in psych_df.columns if col.startswith('keyword_count_')],
            
            # Text statistics
            'text_length', 'word_count', 'avg_word_length', 'sentence_count',
            
            # Medication features
            *[col for col in psych_df.columns if col.startswith('med_')],
            'num_medications', 'polypharmacy',
            
            # Alignment features
            *[col for col in psych_df.columns if col.startswith('alignment_')],
            
            # Diagnosis features
            'num_psych_diagnoses', 'num_diagnosis_groups', 'num_cluster_categories'
        ]
        
        # Filter to existing columns
        feature_cols = [col for col in feature_cols if col in psych_df.columns]
        
        logger.info(f"Created feature matrix with {len(feature_cols)} features")
        
        return psych_df
    
    def process(self, psych_df: pd.DataFrame) -> pd.DataFrame:
        """Run full feature engineering pipeline.
        
        Args:
            psych_df: Psychiatric patients DataFrame
            
        Returns:
            DataFrame with extracted features
        """
        logger.info("Starting feature engineering")
        
        # Extract text features
        psych_df = self.extract_keyword_features(psych_df)
        psych_df = self.extract_text_statistics(psych_df)
        
        # Extract medication features
        psych_df = self.extract_medication_features(psych_df)
        psych_df = self.extract_medication_keyword_alignment(psych_df)
        
        # Extract diagnosis features
        psych_df = self.extract_diagnosis_features(psych_df)
        
        # Create feature matrix
        psych_df = self.create_feature_matrix(psych_df)
        
        logger.info("Feature engineering complete")
        
        return psych_df
