"""Module 1: Data Loading and Preprocessing

This module handles loading MIMIC-III CSV files, merging them,
and creating a unified patient-level dataset.
"""

import pandas as pd
import logging
from typing import Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MIMICDataLoader:
    """Load and merge MIMIC-III dataset files."""
    
    def __init__(self, config: Dict):
        """Initialize data loader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config['data']
        self.processing_config = config['processing']
        
        self.notes_path = self.data_config['notes_path']
        self.diagnoses_path = self.data_config['diagnoses_path']
        self.prescriptions_path = self.data_config['prescriptions_path']
        
        self.chunk_size = self.processing_config.get('chunk_size', 5000)
        
    def load_diagnoses(self) -> pd.DataFrame:
        """Load DIAGNOSES_ICD file.
        
        Returns:
            DataFrame with diagnoses data
        """
        logger.info(f"Loading diagnoses from {self.diagnoses_path}")
        
        try:
            # Load in chunks if file is large
            chunks = []
            for chunk in tqdm(pd.read_csv(self.diagnoses_path, chunksize=self.chunk_size),
                            desc="Loading diagnoses"):
                chunks.append(chunk)
            
            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"Loaded {len(df)} diagnosis records")
            
            # Handle missing values
            df['ICD9_CODE'] = df['ICD9_CODE'].fillna('UNKNOWN')
            
            return df
            
        except FileNotFoundError:
            logger.error(f"Diagnoses file not found: {self.diagnoses_path}")
            logger.info("Creating empty DataFrame - please ensure data files are accessible")
            return pd.DataFrame(columns=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE'])
        except Exception as e:
            logger.error(f"Error loading diagnoses: {e}")
            raise
    
    def load_noteevents(self) -> pd.DataFrame:
        """Load NOTEEVENTS file.
        
        Returns:
            DataFrame with clinical notes
        """
        logger.info(f"Loading notes from {self.notes_path}")
        
        try:
            chunks = []
            for chunk in tqdm(pd.read_csv(self.notes_path, chunksize=self.chunk_size),
                            desc="Loading notes"):
                chunks.append(chunk)
            
            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"Loaded {len(df)} note records")
            
            # Handle missing values in text
            df['TEXT'] = df['TEXT'].fillna('')
            
            return df
            
        except FileNotFoundError:
            logger.error(f"Notes file not found: {self.notes_path}")
            logger.info("Creating empty DataFrame - please ensure data files are accessible")
            return pd.DataFrame(columns=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 
                                        'CHARTTIME', 'STORETIME', 'CATEGORY', 'DESCRIPTION', 
                                        'CGID', 'ISERROR', 'TEXT'])
        except Exception as e:
            logger.error(f"Error loading notes: {e}")
            raise
    
    def load_prescriptions(self) -> pd.DataFrame:
        """Load PRESCRIPTIONS file.
        
        Returns:
            DataFrame with prescription data
        """
        logger.info(f"Loading prescriptions from {self.prescriptions_path}")
        
        try:
            chunks = []
            for chunk in tqdm(pd.read_csv(self.prescriptions_path, chunksize=self.chunk_size),
                            desc="Loading prescriptions"):
                chunks.append(chunk)
            
            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"Loaded {len(df)} prescription records")
            
            # Handle missing drug names
            df['DRUG'] = df['DRUG'].fillna('UNKNOWN')
            
            return df
            
        except FileNotFoundError:
            logger.error(f"Prescriptions file not found: {self.prescriptions_path}")
            logger.info("Creating empty DataFrame - please ensure data files are accessible")
            return pd.DataFrame(columns=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 
                                        'STARTDATE', 'ENDDATE', 'DRUG_TYPE', 'DRUG', 
                                        'DRUG_NAME_POE', 'DRUG_NAME_GENERIC'])
        except Exception as e:
            logger.error(f"Error loading prescriptions: {e}")
            raise
    
    def filter_discharge_summaries(self, notes_df: pd.DataFrame) -> pd.DataFrame:
        """Filter for discharge summaries only.
        
        Args:
            notes_df: Notes DataFrame
            
        Returns:
            Filtered DataFrame with discharge summaries
        """
        logger.info("Filtering discharge summaries")
        
        # Filter for discharge summaries
        discharge_df = notes_df[notes_df['CATEGORY'] == 'Discharge summary'].copy()
        
        # Filter by minimum length
        min_length = self.processing_config.get('min_note_length', 100)
        discharge_df['text_length'] = discharge_df['TEXT'].str.len()
        discharge_df = discharge_df[discharge_df['text_length'] >= min_length]
        
        logger.info(f"Found {len(discharge_df)} discharge summaries with length >= {min_length}")
        
        return discharge_df
    
    def merge_datasets(self, 
                      diagnoses_df: pd.DataFrame,
                      notes_df: pd.DataFrame,
                      prescriptions_df: pd.DataFrame) -> pd.DataFrame:
        """Merge all datasets on SUBJECT_ID and HADM_ID.
        
        Args:
            diagnoses_df: Diagnoses DataFrame
            notes_df: Notes DataFrame  
            prescriptions_df: Prescriptions DataFrame
            
        Returns:
            Merged DataFrame
        """
        logger.info("Merging datasets")
        
        # Start with discharge summaries
        discharge_df = self.filter_discharge_summaries(notes_df)
        
        # Merge with diagnoses
        merged_df = discharge_df.merge(
            diagnoses_df,
            on=['SUBJECT_ID', 'HADM_ID'],
            how='left',
            suffixes=('_note', '_diag')
        )
        
        # Merge with prescriptions
        merged_df = merged_df.merge(
            prescriptions_df,
            on=['SUBJECT_ID', 'HADM_ID'],
            how='left',
            suffixes=('', '_presc')
        )
        
        logger.info(f"Merged dataset has {len(merged_df)} records")
        logger.info(f"Unique patients: {merged_df['SUBJECT_ID'].nunique()}")
        logger.info(f"Unique admissions: {merged_df['HADM_ID'].nunique()}")
        
        return merged_df
    
    def create_patient_level_dataset(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data at patient-admission level.
        
        Args:
            merged_df: Merged DataFrame
            
        Returns:
            Patient-level aggregated DataFrame
        """
        logger.info("Creating patient-level dataset")
        
        # Group by SUBJECT_ID and HADM_ID
        grouped = merged_df.groupby(['SUBJECT_ID', 'HADM_ID'])
        
        patient_data = []
        
        for (subject_id, hadm_id), group in tqdm(grouped, desc="Aggregating patients"):
            # Get discharge summary (take first if multiple)
            discharge_text = group['TEXT'].iloc[0] if len(group) > 0 else ''
            
            # Aggregate diagnoses
            diagnoses = group['ICD9_CODE'].dropna().unique().tolist()
            
            # Aggregate medications
            medications = group['DRUG'].dropna().unique().tolist()
            
            patient_data.append({
                'SUBJECT_ID': subject_id,
                'HADM_ID': hadm_id,
                'discharge_summary': discharge_text,
                'diagnoses': diagnoses,
                'medications': medications,
                'num_diagnoses': len(diagnoses),
                'num_medications': len(medications)
            })
        
        patient_df = pd.DataFrame(patient_data)
        logger.info(f"Created patient-level dataset with {len(patient_df)} admissions")
        
        return patient_df
    
    def load_all(self) -> pd.DataFrame:
        """Load all data and create unified dataset.
        
        Returns:
            Unified patient-level DataFrame
        """
        logger.info("Starting data loading pipeline")
        
        # Load individual files
        diagnoses_df = self.load_diagnoses()
        notes_df = self.load_noteevents()
        prescriptions_df = self.load_prescriptions()
        
        # Merge datasets
        merged_df = self.merge_datasets(diagnoses_df, notes_df, prescriptions_df)
        
        # Create patient-level dataset
        patient_df = self.create_patient_level_dataset(merged_df)
        
        # Save merged dataset
        output_path = self.data_config['merged_data_path']
        patient_df.to_csv(output_path, index=False)
        logger.info(f"Saved merged dataset to {output_path}")
        
        return patient_df
