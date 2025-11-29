# Getting Started with MIMIC-III Psychiatric Clustering Pipeline

This guide will help you get started with the psychiatric disorder clustering pipeline.

## 🚀 Quick Start (5 Minutes)

### Option 1: Run Full Pipeline

```bash
# Navigate to project directory
cd /app

# Run the complete pipeline
python main_pipeline.py
```

**Expected Output:**
- Processed psychiatric cohort
- ClinicalBERT embeddings
- Cluster assignments
- Characterization reports
- Publication-ready visualizations

**Runtime**: 1-3 hours (depending on dataset size and CPU speed)

### Option 2: Interactive Jupyter Notebooks

```bash
# Start Jupyter
cd /app
jupyter notebook

# Open notebooks in order:
# 01_data_exploration.ipynb
# 02_feature_engineering.ipynb
# 03_embeddings.ipynb (create if needed)
# 04_clustering.ipynb (create if needed)
# 05_final_analysis.ipynb (create if needed)
```

**Benefits:**
- Step-by-step exploration
- Interactive visualizations
- Ability to modify and experiment
- Better understanding of each stage

## 📋 Prerequisites Checklist

Before running the pipeline, ensure you have:

### ✅ Data Requirements
- [ ] MIMIC-III dataset access (PhysioNet credentialed)
- [ ] Three CSV files:
  - `NOTEEVENTS_sorted.csv`
  - `DIAGNOSES_ICD_sorted.csv`
  - `PRESCRIPTIONS_sorted.csv`
- [ ] Files accessible at paths specified in `config/config.yaml`

### ✅ System Requirements
- [ ] Python 3.8 or higher
- [ ] 8GB+ RAM (16GB recommended)
- [ ] 5GB+ free disk space
- [ ] Internet connection (for first-time model download)

### ✅ Software Requirements
- [ ] All Python packages installed (`pip install -r requirements_ml.txt`)
- [ ] Jupyter notebook (if using interactive mode)

## 🔧 Setup Instructions

### Step 1: Verify Installation

```bash
# Check Python version
python --version  # Should be 3.8+

# Verify key packages
python -c \"import pandas, torch, transformers; print('All packages installed!')\"
```

### Step 2: Configure Paths

Edit `config/config.yaml`:

```yaml
data:
  notes_path: \"YOUR_PATH/NOTEEVENTS_sorted.csv\"
  diagnoses_path: \"YOUR_PATH/DIAGNOSES_ICD_sorted.csv\"
  prescriptions_path: \"YOUR_PATH/PRESCRIPTIONS_sorted.csv\"
```

**Windows paths**: Use double backslashes `\\\\` or forward slashes `/`

```yaml
# Windows example
notes_path: \"C:\\\\Users\\\\YourName\\\\Data\\\\NOTEEVENTS_sorted.csv\"

# Or use forward slashes
notes_path: \"C:/Users/YourName/Data/NOTEEVENTS_sorted.csv\"
```

### Step 3: Test Data Access

```python
# Quick test script
import pandas as pd

# Try loading a small chunk
notes_path = \"YOUR_PATH/NOTEEVENTS_sorted.csv\"
df = pd.read_csv(notes_path, nrows=10)
print(f\"Successfully loaded {len(df)} rows\")
print(df.columns.tolist())
```

If this works, you're ready to run the pipeline!

### Step 4: Create Output Directories

```bash
mkdir -p /app/outputs/{embeddings,clusters,visualizations,reports}
```

## 📊 Understanding the Pipeline

### Pipeline Stages

```
1. Data Loading (5-10 min)
   ├─ Load CSV files
   ├─ Merge datasets
   └─ Create patient-level data

2. Cohort Selection (2-5 min)
   ├─ Filter psychiatric patients
   ├─ Extract diagnoses
   └─ Create framework flags

3. Feature Engineering (5-15 min)
   ├─ Extract keywords
   ├─ Categorize medications
   └─ Engineer diagnosis features

4. Embedding Generation (30-90 min) ⏱️ LONGEST STEP
   ├─ Load ClinicalBERT
   ├─ Generate embeddings
   └─ Reduce dimensionality

5. Clustering (10-20 min)
   ├─ Try multiple algorithms
   ├─ Select best model
   └─ Assign clusters

6. Characterization (5-10 min)
   ├─ Analyze diagnoses
   ├─ Analyze medications
   └─ Extract keywords

7. Visualization (10-15 min)
   ├─ Create plots
   ├─ Generate heatmaps
   └─ Make word clouds

8. Validation (5-10 min)
   ├─ Calculate metrics
   ├─ Assess stability
   └─ Generate reports
```

**Total Time**: 1-3 hours

### Key Configuration Parameters

#### For Faster Testing

```yaml
processing:
  sample_size: 1000  # Test with 1000 patients

embeddings:
  batch_size: 4  # Smaller batches
  dimensionality_reduction:
    n_components: 30  # Fewer dimensions
```

#### For Full Production Run

```yaml
processing:
  sample_size: null  # Use all data

embeddings:
  batch_size: 8  # Larger batches (if memory allows)
  dimensionality_reduction:
    n_components: 50  # More dimensions for better quality
```

## 🎯 What to Expect: Normal Results

### Cohort Size
- **Expected**: 15-30% of total patients are psychiatric
- **Example**: From 10,000 patients → 1,500-3,000 psychiatric

### Clustering
- **Target**: 6 clusters (based on framework)
- **Actual**: 4-8 clusters is normal (data-driven)
- **Silhouette Score**: 0.3-0.5 is good for clinical data

### Top Diagnoses per Cluster
- Each cluster should have 3-5 dominant ICD-9 codes
- Some overlap between clusters is normal (comorbidity)

### Medication Patterns
- Clear medication class preferences per cluster
- Example: Cluster with antipsychotics → psychotic disorders

## ⚠️ Common Issues and Solutions

### Issue 1: \"File not found\"
**Cause**: Incorrect paths in config.yaml
**Solution**:
```bash
# Verify file exists
ls -la \"YOUR_PATH/NOTEEVENTS_sorted.csv\"

# Update config with correct path
# Use absolute paths (full path from root)
```

### Issue 2: \"Out of memory\"
**Cause**: Dataset too large for available RAM
**Solution**:
```yaml
# Option A: Use sample
processing:
  sample_size: 1000

# Option B: Reduce batch size
embeddings:
  batch_size: 2

# Option C: Disable dimensionality reduction temporarily
embeddings:
  dimensionality_reduction:
    method: \"none\"
```

### Issue 3: \"ClinicalBERT download fails\"
**Cause**: Network issues or Hugging Face access
**Solution**:
```bash
# Pre-download model
python -c \"from transformers import AutoModel; AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')\"

# Or use alternative model
embeddings:
  model_name: \"bert-base-uncased\"  # Less specialized but works
```

### Issue 4: \"Clustering finds only 1-2 clusters\"
**Cause**: Insufficient data variation or poor features
**Solution**:
- Check if psychiatric filtering removed too many patients
- Try different clustering algorithm (GMM often better)
- Verify feature engineering completed successfully

### Issue 5: \"Pipeline very slow (>5 hours)\"
**Cause**: Large dataset + CPU processing
**Solution**:
```yaml
# Reduce embedding batch processing
embeddings:
  batch_size: 4
  max_length: 256  # Shorter sequences

# Test with sample first
processing:
  sample_size: 500  # Test run
```

## 📈 Monitoring Progress

### Check Logs

```bash
# View real-time progress
tail -f /app/outputs/pipeline.log

# Search for errors
grep ERROR /app/outputs/pipeline.log
```

### Monitor Resource Usage

```bash
# Check memory usage
free -h

# Check disk space
df -h /app/outputs
```

### Intermediate Checkpoints

The pipeline saves intermediate results:
```bash
/app/outputs/merged_psychiatric_dataset.csv  # After data loading
/app/outputs/processed_psychiatric_cohort.csv  # After feature engineering
/app/outputs/embeddings/clinical_bert_embeddings.pkl  # After embedding
```

You can resume from checkpoints if needed.

## 🎓 Next Steps After Pipeline Completes

### 1. Review Visualizations

```bash
# Open visualization directory
cd /app/outputs/visualizations

# View images (depends on your system)
# On Linux with GUI:
xdg-open clusters_tsne_2d.png

# Or copy to your local machine
```

### 2. Analyze Reports

```bash
# View cluster statistics
cat /app/outputs/reports/cluster_statistics.csv

# View top diagnoses
cat /app/outputs/reports/cluster_diagnoses.csv

# View framework alignment
cat /app/outputs/reports/framework_comparison.csv
```

### 3. Interactive Exploration

```python
# In Python or Jupyter
import pandas as pd

# Load cluster assignments
clusters = pd.read_csv('/app/outputs/clusters/cluster_assignments.csv')

# Load statistics
stats = pd.read_csv('/app/outputs/reports/cluster_statistics.csv')

# Explore
print(clusters.cluster.value_counts())
print(stats)
```

### 4. Clinical Interpretation

For each cluster, consider:
- **Dominant diagnoses**: What's the primary disorder category?
- **Medication patterns**: Do prescriptions align with diagnoses?
- **Keywords**: Do clinical notes reflect expected symptoms?
- **Comorbidity**: High or low overlap with other categories?

### 5. Publish Results

Outputs are publication-ready:
- Figures: 300 DPI PNG format
- Tables: CSV format (import to Excel/LaTeX)
- Reports: Structured for manuscript methods/results sections

## 🔄 Running Multiple Experiments

### Experiment 1: Different Number of Clusters

```yaml
# config.yaml
clustering:
  n_clusters: 4  # Try 4, 5, 6, 7, 8
```

```bash
python main_pipeline.py
# Results saved to outputs/
```

### Experiment 2: Different Algorithms

```yaml
clustering:
  algorithms: [\"kmeans\"]  # Test one at a time
  # or [\"hierarchical\"]
  # or [\"gmm\"]
```

### Experiment 3: Feature Variations

Try removing feature types to test impact:
- Keywords only
- Medications only  
- Diagnoses only
- All features combined

## 📚 Additional Resources

### Documentation
- Full documentation: `README_MIMIC.md`
- Notebook guide: `notebooks/README.md`
- Module details: See docstrings in `/app/src/`

### MIMIC-III Resources
- PhysioNet: https://physionet.org/content/mimiciii/
- Documentation: https://mimic.mit.edu/
- Tutorials: https://github.com/MIT-LCP/mimic-code

### ClinicalBERT
- Paper: https://arxiv.org/abs/1904.05342
- Model: https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT

## 💡 Tips for Success

1. **Start Small**: Test with sample_size: 1000 first
2. **Monitor Logs**: Watch pipeline.log for issues
3. **Save Configs**: Keep track of config.yaml versions
4. **Document Changes**: Note any modifications made
5. **Backup Results**: Copy outputs/ directory after completion

## 🆘 Getting Help

If you encounter issues:

1. Check `outputs/pipeline.log` for error messages
2. Review troubleshooting section above
3. Verify all prerequisites are met
4. Try with smaller sample size first
5. Check that data files are accessible

## ✨ You're Ready!

You now have everything you need to run the psychiatric clustering pipeline. Start with:

```bash
python main_pipeline.py
```

And watch the magic happen! 🚀

Good luck with your research! 🎓
