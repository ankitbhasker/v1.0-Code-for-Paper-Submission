"""Module 2: Psychiatric Cohort Selection

This module filters for psychiatric admissions and creates
psychiatric diagnosis flags.
"""

import pandas as pd
import logging
from typing import Dict, List
from tqdm import tqdm
import sys
sys.path.append('/app')
from src.utils import is_psychiatric_icd9

logger = logging.getLogger(__name__)


class PsychiatricCohortSelector:
    """Select and preprocess psychiatric patient cohort."""
    
    def __init__(self, config: Dict):
        """Initialize cohort selector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.processing_config = config['processing']
        self.cluster_framework = config['cluster_framework']
        
        # Psychiatric ICD-9 range
        psych_range = self.processing_config['psychiatric_icd9_range']
        self.min_psych_code = psych_range[0]
        self.max_psych_code = psych_range[1]
    
    def identify_psychiatric_patients(self, patient_df: pd.DataFrame) -> pd.DataFrame:
        """Identify patients with psychiatric diagnoses.
        
        Args:
            patient_df: Patient-level DataFrame
            
        Returns:
            Filtered DataFrame with only psychiatric patients
        """
        logger.info("Identifying psychiatric patients")
        
        def has_psychiatric_diagnosis(diagnoses: List[str]) -> bool:
            """Check if patient has any psychiatric diagnosis."""
            if not isinstance(diagnoses, list):
                return False
            
            for diag in diagnoses:
                if is_psychiatric_icd9(diag, self.min_psych_code, self.max_psych_code):
                    return True
            return False
        
        # Apply filter
        patient_df['is_psychiatric'] = patient_df['diagnoses'].apply(has_psychiatric_diagnosis)
        
        psych_df = patient_df[patient_df['is_psychiatric']].copy()
        
        logger.info(f"Found {len(psych_df)} psychiatric admissions")
        logger.info(f"({len(psych_df) / len(patient_df) * 100:.1f}% of total)")
        
        return psych_df
    
    def extract_primary_diagnoses(self, psych_df: pd.DataFrame) -> pd.DataFrame:
        """Extract primary psychiatric diagnosis for each patient.
        
        Args:
            psych_df: Psychiatric patients DataFrame
            
        Returns:
            DataFrame with primary diagnosis column
        """
        logger.info("Extracting primary psychiatric diagnoses")
        
        def get_primary_psychiatric_diagnosis(diagnoses: List[str]) -> str:
            """Get first psychiatric diagnosis."""
            if not isinstance(diagnoses, list):
                return None
            
            for diag in diagnoses:
                if is_psychiatric_icd9(diag, self.min_psych_code, self.max_psych_code):
                    return diag
            return None
        
        psych_df['primary_psych_diagnosis'] = psych_df['diagnoses'].apply(
            get_primary_psychiatric_diagnosis
        )
        
        # Get all psychiatric diagnoses
        def get_all_psychiatric_diagnoses(diagnoses: List[str]) -> List[str]:
            """Get all psychiatric diagnoses."""
            if not isinstance(diagnoses, list):
                return []
            
            return [d for d in diagnoses 
                   if is_psychiatric_icd9(d, self.min_psych_code, self.max_psych_code)]
        
        psych_df['all_psych_diagnoses'] = psych_df['diagnoses'].apply(
            get_all_psychiatric_diagnoses
        )
        
        psych_df['num_psych_diagnoses'] = psych_df['all_psych_diagnoses'].apply(len)
        
        logger.info(f"Average psychiatric diagnoses per patient: "
                   f"{psych_df['num_psych_diagnoses'].mean():.2f}")
        
        return psych_df
    
    def create_cluster_flags(self, psych_df: pd.DataFrame) -> pd.DataFrame:
        """Create binary flags for each theoretical cluster category.
        
        Args:
            psych_df: Psychiatric patients DataFrame
            
        Returns:
            DataFrame with cluster flags
        """
        logger.info("Creating cluster category flags")
        
        for cluster_name, cluster_info in self.cluster_framework.items():
            flag_name = f'flag_{cluster_name}'
            icd9_prefixes = cluster_info['icd9_codes']
            
            def has_cluster_diagnosis(diagnoses: List[str]) -> bool:
                """Check if patient has diagnosis in this cluster."""
                if not isinstance(diagnoses, list):
                    return False
                
                for diag in diagnoses:
                    diag_str = str(diag)
                    for prefix in icd9_prefixes:
                        if diag_str.startswith(str(prefix)):
                            return True
                return False
            
            psych_df[flag_name] = psych_df['all_psych_diagnoses'].apply(has_cluster_diagnosis)
            
            count = psych_df[flag_name].sum()
            logger.info(f"  {cluster_info['name']}: {count} patients ({count/len(psych_df)*100:.1f}%)")
        
        return psych_df
    
    def calculate_comorbidity(self, psych_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate psychiatric comorbidity indicators.
        
        Args:
            psych_df: Psychiatric patients DataFrame
            
        Returns:
            DataFrame with comorbidity features
        """
        logger.info("Calculating comorbidity indicators")
        
        # Count how many cluster categories each patient belongs to
        cluster_flags = [col for col in psych_df.columns if col.startswith('flag_')]
        psych_df['num_cluster_categories'] = psych_df[cluster_flags].sum(axis=1)
        
        # Binary comorbidity flag
        psych_df['has_comorbidity'] = (psych_df['num_cluster_categories'] > 1).astype(int)
        
        comorbid_count = psych_df['has_comorbidity'].sum()
        logger.info(f"Patients with comorbidity: {comorbid_count} "
                   f"({comorbid_count/len(psych_df)*100:.1f}%)")
        
        return psych_df
    
    def process(self, patient_df: pd.DataFrame) -> pd.DataFrame:
        """Run full preprocessing pipeline.
        
        Args:
            patient_df: Patient-level DataFrame
            
        Returns:
            Preprocessed psychiatric cohort DataFrame
        """
        logger.info("Starting psychiatric cohort selection")
        
        # Identify psychiatric patients
        psych_df = self.identify_psychiatric_patients(patient_df)
        
        # Extract primary diagnoses
        psych_df = self.extract_primary_diagnoses(psych_df)
        
        # Create cluster flags
        psych_df = self.create_cluster_flags(psych_df)
        
        # Calculate comorbidity
        psych_df = self.calculate_comorbidity(psych_df)
        
        logger.info(f"Preprocessing complete. Final cohort: {len(psych_df)} patients")
        
        return psych_df
