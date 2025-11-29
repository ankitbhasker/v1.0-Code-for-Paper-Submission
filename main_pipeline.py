#!/usr/bin/env python3
"""Main Pipeline for MIMIC-III Psychiatric Clustering

This script orchestrates the entire clustering pipeline from
data loading to final validation and visualization.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.utils import (
    load_config, 
    setup_logging, 
    set_random_seeds,
    ensure_dir
)
from src.data_loader import MIMICDataLoader
from src.preprocessor import PsychiatricCohortSelector
from src.feature_engineer import ClinicalFeatureEngineer
from src.embedder import ClinicalBERTEmbedder
from src.clustering import PsychiatricClusterer
from src.characterization import ClusterCharacterizer
from src.visualizer import ClusterVisualizer
from src.validation import ClusterValidator


def main():
    """Run the complete psychiatric clustering pipeline."""
    
    print("="*80)
    print("MIMIC-III PSYCHIATRIC DISORDER CLUSTERING PIPELINE")
    print("="*80)
    print()
    
    # Load configuration
    print("[1/9] Loading configuration...")
    config = load_config('config/config.yaml')
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("Pipeline started")
    
    # Set random seeds
    set_random_seeds(config.get('random_state', 42))
    
    # Ensure output directories exist
    ensure_dir(config['data']['output_dir'])
    ensure_dir(f"{config['data']['output_dir']}/embeddings")
    ensure_dir(f"{config['data']['output_dir']}/clusters")
    ensure_dir(f"{config['data']['output_dir']}/visualizations")
    ensure_dir(f"{config['data']['output_dir']}/reports")
    
    try:
        # Module 1: Data Loading
        print("\n[2/9] Loading and merging MIMIC-III data...")
        data_loader = MIMICDataLoader(config)
        patient_df = data_loader.load_all()
        print(f"  ✓ Loaded {len(patient_df)} patient admissions")
        
        # Module 2: Psychiatric Cohort Selection
        print("\n[3/9] Selecting psychiatric cohort...")
        preprocessor = PsychiatricCohortSelector(config)
        psych_df = preprocessor.process(patient_df)
        print(f"  ✓ Identified {len(psych_df)} psychiatric admissions")
        
        # Module 3: Feature Engineering
        print("\n[4/9] Engineering clinical features...")
        feature_engineer = ClinicalFeatureEngineer(config)
        psych_df = feature_engineer.process(psych_df)
        print(f"  ✓ Extracted features from notes, medications, and diagnoses")
        
        # Save processed data
        processed_path = f"{config['data']['output_dir']}/processed_psychiatric_cohort.csv"
        psych_df.to_csv(processed_path, index=False)
        print(f"  ✓ Saved processed data to {processed_path}")
        
        # Module 4: ClinicalBERT Embeddings
        print("\n[5/9] Generating ClinicalBERT embeddings...")
        print("  (This may take a while depending on dataset size and CPU speed)")
        embedder = ClinicalBERTEmbedder(config)
        psych_df, embeddings = embedder.process(psych_df)
        print(f"  ✓ Generated embeddings with shape {embeddings.shape}")
        
        # Module 5: Clustering
        print("\n[6/9] Performing clustering analysis...")
        clusterer = PsychiatricClusterer(config)
        labels, comparison_df = clusterer.process(embeddings)
        print(f"  ✓ Identified {len(set(labels) - {-1})} clusters")
        print(f"  ✓ Best algorithm: {clusterer.best_algorithm}")
        
        # Save cluster assignments
        psych_df['cluster'] = labels
        clusters_path = config['data']['clusters_path']
        psych_df[['SUBJECT_ID', 'HADM_ID', 'cluster']].to_csv(clusters_path, index=False)
        print(f"  ✓ Saved cluster assignments to {clusters_path}")
        
        # Module 6: Cluster Characterization
        print("\n[7/9] Characterizing clusters...")
        characterizer = ClusterCharacterizer(config)
        characterization_report = characterizer.process(psych_df, labels)
        print(f"  ✓ Generated profiles for all clusters")
        
        # Module 7: Visualization
        print("\n[8/9] Creating visualizations...")
        visualizer = ClusterVisualizer(config)
        visualizer.process(embeddings, labels, characterization_report)
        print(f"  ✓ Generated all visualizations")
        
        # Module 8: Validation
        print("\n[9/9] Validating clustering results...")
        validator = ClusterValidator(config)
        validation_report = validator.process(embeddings, labels, psych_df)
        print(f"  ✓ Completed validation analysis")
        
        # Print summary
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nSummary:")
        print(f"  • Total patients analyzed: {len(psych_df)}")
        print(f"  • Number of clusters: {len(set(labels) - {-1})}")
        print(f"  • Clustering algorithm: {clusterer.best_algorithm}")
        print(f"  • Silhouette score: {clusterer.best_score:.3f}")
        
        if 'internal_metrics' in validation_report:
            metrics = validation_report['internal_metrics']
            print(f"  • Davies-Bouldin score: {metrics.get('davies_bouldin_score', 'N/A')}")
            print(f"  • Calinski-Harabasz score: {metrics.get('calinski_harabasz_score', 'N/A')}")
        
        print(f"\nOutputs saved to: {config['data']['output_dir']}/")
        print("  • Processed data: processed_psychiatric_cohort.csv")
        print("  • Embeddings: embeddings/clinical_bert_embeddings.pkl")
        print("  • Cluster assignments: clusters/cluster_assignments.csv")
        print("  • Reports: reports/")
        print("  • Visualizations: visualizations/")
        
        print("\nNext steps:")
        print("  1. Review visualizations in outputs/visualizations/")
        print("  2. Examine cluster profiles in outputs/reports/")
        print("  3. Use Jupyter notebooks for interactive analysis")
        print("\n" + "="*80)
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
