# MIMIC-III Psychiatric Disorder Clustering Pipeline

A comprehensive machine learning pipeline for identifying latent psychiatric disorder subtypes using unsupervised learning on the MIMIC-III clinical database.

## 📋 Overview

This project implements an end-to-end pipeline that:
- Processes MIMIC-III clinical data (discharge summaries, diagnoses, prescriptions)
- Filters and preprocesses psychiatric patient cohorts
- Extracts clinical features aligned with theoretical psychiatric frameworks
- Generates ClinicalBERT embeddings for clinical notes
- Applies multiple clustering algorithms to identify psychiatric subtypes
- Characterizes clusters with symptoms, medications, and ICD-9 patterns
- Validates results through internal metrics and stability analysis
- Produces publication-ready visualizations

## 🎯 Research Objectives

1. **Data Integration**: Merge NOTEEVENTS, DIAGNOSES_ICD, and PRESCRIPTIONS
2. **Psychiatric Focus**: Filter for psychiatric discharge summaries (ICD-9: 290-319)
3. **Feature Engineering**: Extract multi-modal clinical features
4. **Text Embedding**: Encode clinical notes using ClinicalBERT
5. **Clustering**: Apply unsupervised learning (K-Means, Hierarchical, DBSCAN, GMM)
6. **Characterization**: Profile clusters with clinical indicators
7. **Validation**: Assess cluster validity and interpretability

## 🏗️ Project Structure

```
/app/
├── config/
│   └── config.yaml              # Pipeline configuration
├── data/                        # Dataset storage (place CSV files here)
├── src/                         # Source code modules
│   ├── __init__.py
│   ├── utils.py                 # Utility functions
│   ├── data_loader.py          # Module 1: Data loading
│   ├── preprocessor.py         # Module 2: Cohort selection
│   ├── feature_engineer.py     # Module 3: Feature extraction
│   ├── embedder.py             # Module 4: ClinicalBERT embeddings
│   ├── clustering.py           # Module 5: Clustering algorithms
│   ├── characterization.py     # Module 6: Cluster profiling
│   ├── visualizer.py           # Module 7: Visualization
│   └── validation.py           # Module 8: Validation
├── notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_embeddings.ipynb
│   ├── 04_clustering.ipynb
│   └── 05_final_analysis.ipynb
├── outputs/                     # Generated outputs
│   ├── embeddings/             # Saved embeddings
│   ├── clusters/               # Cluster assignments
│   ├── visualizations/         # Plots and figures
│   └── reports/                # Analysis reports
├── main_pipeline.py            # Main orchestration script
├── requirements_ml.txt         # Python dependencies
└── README_MIMIC.md            # This file
```

## 📦 Installation

### 1. Install Dependencies

```bash
# Install Python packages
pip install -r requirements_ml.txt

# Download spaCy language model (optional)
python -m spacy download en_core_web_sm
```

### 2. Download ClinicalBERT Model

The pipeline will automatically download ClinicalBERT on first run. Ensure you have:
- Active internet connection
- ~500MB free disk space
- Hugging Face transformers library installed

## 🚀 Usage

### Quick Start: Run Full Pipeline

```bash
# Run complete pipeline
python main_pipeline.py
```

The pipeline will:
1. Load and merge MIMIC-III CSV files
2. Select psychiatric cohort
3. Engineer clinical features
4. Generate ClinicalBERT embeddings
5. Perform clustering analysis
6. Characterize and profile clusters
7. Create visualizations
8. Validate results

**Expected runtime**: 1-3 hours depending on dataset size and CPU speed

### Interactive Analysis: Jupyter Notebooks

```bash
# Launch Jupyter
jupyter notebook

# Navigate to notebooks/ directory
# Open notebooks in sequence:
# 01_data_exploration.ipynb -> ... -> 05_final_analysis.ipynb
```

### Running Individual Modules

```python
from src.utils import load_config
from src.data_loader import MIMICDataLoader

# Load configuration
config = load_config('config/config.yaml')

# Run specific module
data_loader = MIMICDataLoader(config)
patient_df = data_loader.load_all()
```

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

### Dataset Paths
```yaml
data:
  notes_path: "N:\\M.Tech Thesis\\Dataset\\...\\NOTEEVENTS_sorted.csv"
  diagnoses_path: "N:\\M.Tech Thesis\\Dataset\\...\\DIAGNOSES_ICD_sorted.csv"
  prescriptions_path: "N:\\M.Tech Thesis\\Dataset\\...\\PRESCRIPTIONS_sorted.csv"
```

### ClinicalBERT Settings
```yaml
embeddings:
  model_name: "emilyalsentzer/Bio_ClinicalBERT"
  max_length: 512
  batch_size: 8
  device: "cpu"  # Change to "cuda" if GPU available
```

### Clustering Parameters
```yaml
clustering:
  algorithms: ["kmeans", "hierarchical", "dbscan", "gmm"]
  n_clusters: 6  # Target number based on framework
```

## 📊 Theoretical Framework: Target Clusters

### 1. Internalizing Disorders
- **ICD-9**: 296.xx (mood), 300.xx (anxiety), 311 (depression)
- **Medications**: SSRIs, SNRIs, anxiolytics
- **Keywords**: depressed, anxious, worry, hopeless

### 2. Externalizing Disorders
- **ICD-9**: 303.xx-305.xx (substance use), 312.xx (conduct)
- **Medications**: Naltrexone, buprenorphine, methadone
- **Keywords**: alcohol, cocaine, substance, impulsive

### 3. Psychotic Disorders
- **ICD-9**: 295.xx (schizophrenia), 297.xx (delusional)
- **Medications**: Antipsychotics (risperidone, olanzapine, haloperidol)
- **Keywords**: hallucination, delusion, psychotic, paranoid

### 4. Neurodevelopmental Disorders
- **ICD-9**: 299.xx (autism), 314.xx (ADHD)
- **Medications**: Stimulants (methylphenidate, amphetamine)
- **Keywords**: adhd, attention, hyperactive, developmental

### 5. Somatic-Focused Disorders
- **ICD-9**: 300.7x (somatoform), 307.8x (pain disorder)
- **Medications**: Gabapentin, pregabalin, duloxetine
- **Keywords**: pain, somatic, chronic pain, fibromyalgia

### 6. Personality Dysfunction
- **ICD-9**: 301.xx (personality disorders), 301.83 (borderline)
- **Medications**: Mood stabilizers (lithium, valproate, lamotrigine)
- **Keywords**: borderline, impulsive, self-harm, unstable

## 📈 Output Files

### Processed Data
- `outputs/merged_psychiatric_dataset.csv` - Patient-level aggregated data
- `outputs/processed_psychiatric_cohort.csv` - Final processed cohort with features

### Embeddings
- `outputs/embeddings/clinical_bert_embeddings.pkl` - ClinicalBERT embeddings

### Cluster Results
- `outputs/clusters/cluster_assignments.csv` - Patient cluster assignments
- `outputs/clusters/algorithm_comparison.csv` - Performance comparison

### Reports
- `outputs/reports/cluster_diagnoses.csv` - Top ICD-9 codes per cluster
- `outputs/reports/cluster_medications.csv` - Top medications per cluster
- `outputs/reports/cluster_statistics.csv` - Statistical profiles
- `outputs/reports/cluster_keywords.txt` - Characteristic keywords
- `outputs/reports/framework_comparison.csv` - Theoretical alignment
- `outputs/reports/validation_metrics.csv` - Clustering metrics

### Visualizations
- `outputs/visualizations/clusters_tsne_2d.png` - t-SNE cluster plot
- `outputs/visualizations/clusters_umap_2d.png` - UMAP cluster plot
- `outputs/visualizations/cluster_sizes.png` - Cluster size distribution
- `outputs/visualizations/framework_heatmap.png` - Framework alignment
- `outputs/visualizations/medication_profiles.png` - Medication patterns
- `outputs/visualizations/diagnosis_distribution.png` - Diagnosis patterns
- `outputs/visualizations/cluster_statistics.png` - Statistical comparisons
- `outputs/visualizations/cluster_wordclouds.png` - Keyword word clouds
- `outputs/visualizations/elbow_method.png` - Optimal k selection

## 🔬 Methodology

### Feature Engineering
The pipeline extracts three types of features:

1. **Text Features**: 
   - Keyword counts for each cluster category
   - Text length and complexity metrics
   - TF-IDF characteristic terms

2. **Medication Features**:
   - Drug class categorization (SSRI, antipsychotic, stimulant, etc.)
   - Polypharmacy indicators
   - Alignment with cluster-specific medications

3. **Diagnosis Features**:
   - ICD-9 code grouping
   - Comorbidity patterns
   - Primary psychiatric diagnoses

### ClinicalBERT Embedding
- **Model**: Bio_ClinicalBERT (pre-trained on clinical notes)
- **Strategy**: [CLS] token pooling
- **Max Length**: 512 tokens
- **Dimensionality Reduction**: UMAP/PCA to 50 dimensions

### Clustering Algorithms
- **K-Means**: Fast, spherical clusters
- **Hierarchical**: Agglomerative with Ward linkage
- **DBSCAN**: Density-based, finds outliers
- **Gaussian Mixture**: Probabilistic soft clustering

### Validation Metrics
- **Silhouette Score**: Cluster cohesion and separation (0.3+ target)
- **Davies-Bouldin Index**: Cluster separation (lower is better)
- **Calinski-Harabasz Score**: Variance ratio (higher is better)
- **Bootstrap Stability**: Adjusted Rand Index across samples

## 📊 Expected Results

### Success Criteria
- ✅ Silhouette score > 0.3
- ✅ 6 stable clusters identified
- ✅ Each cluster shows distinct clinical profile
- ✅ Meaningful alignment with theoretical framework
- ✅ Interpretable medication and diagnosis patterns

### Typical Cluster Sizes
- Range: 50-500 patients per cluster
- Distribution: Some clusters may be larger (internalizing) vs smaller (neurodevelopmental)
- Outliers: 5-15% noise points expected with DBSCAN

## 🐛 Troubleshooting

### Issue: "File not found" error
**Solution**: Update file paths in `config/config.yaml` to match your system

### Issue: Out of memory during embedding generation
**Solutions**:
- Reduce `batch_size` in config (try 4 or 2)
- Use smaller embedding dimension reduction target
- Process data in chunks

### Issue: ClinicalBERT download fails
**Solutions**:
- Check internet connection
- Try manual download: `from transformers import AutoModel; AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')`
- Use alternative: `bert-base-uncased` (less clinical-specific)

### Issue: Clustering produces only 1-2 clusters
**Solutions**:
- Check if data has sufficient variance
- Try different algorithms (GMM often works better for imbalanced data)
- Adjust DBSCAN eps parameter (try 0.3-0.7 range)

### Issue: Long runtime (>4 hours)
**Solutions**:
- Reduce sample size in config for testing
- Use GPU if available (set device: "cuda")
- Skip dimensionality reduction (set method: "none")

## 📚 Dependencies

### Core Libraries
- pandas >= 2.2.0
- numpy >= 1.26.0
- scikit-learn >= 1.4.0
- torch >= 2.1.0
- transformers >= 4.36.0

### Visualization
- matplotlib >= 3.8.0
- seaborn >= 0.13.0
- plotly >= 5.18.0
- wordcloud >= 1.9.3

### Optional
- umap-learn >= 0.5.5 (recommended for better dimensionality reduction)
- spacy >= 3.7.2 (for advanced text processing)

## 🔬 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{mimic_psych_clustering,
  title={MIMIC-III Psychiatric Disorder Clustering Pipeline},
  author={Your Name},
  year={2025},
  note={Unsupervised learning pipeline for psychiatric subtype discovery}
}
```

## 📄 License

This project uses the MIMIC-III dataset, which requires:
- Completion of CITI "Data or Specimens Only Research" course
- Signed Data Use Agreement
- PhysioNet credentialed access

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📞 Support

For issues or questions:
- Check troubleshooting section above
- Review module-specific logs in `outputs/pipeline.log`
- Open GitHub issue with error details

## 🔄 Version History

### v1.0.0 (Current)
- Initial release
- All 8 modules implemented
- Comprehensive validation and visualization
- CPU-optimized ClinicalBERT inference
- Jupyter notebook interface

---

**Note**: This pipeline is designed for research purposes. Ensure you have appropriate data access permissions and ethical approval before processing real patient data.
