# Jupyter Notebooks for MIMIC-III Psychiatric Clustering

This directory contains interactive Jupyter notebooks for exploring and analyzing the psychiatric clustering pipeline.

## Notebooks Overview

### 01_data_exploration.ipynb
**Purpose**: Initial data loading and exploration
- Load MIMIC-III CSV files
- Merge datasets
- Filter psychiatric cohort
- Visualize data distributions
- Analyze comorbidity patterns

**Runtime**: ~5-10 minutes

### 02_feature_engineering.ipynb
**Purpose**: Extract clinical features
- Keyword extraction from discharge summaries
- Medication categorization
- Diagnosis feature engineering
- Feature correlation analysis

**Runtime**: ~5-15 minutes

### 03_embeddings.ipynb (To be created)
**Purpose**: Generate ClinicalBERT embeddings
- Load ClinicalBERT model
- Generate text embeddings
- Dimensionality reduction
- Visualize embedding space

**Runtime**: ~30-60 minutes (CPU-dependent)

### 04_clustering.ipynb (To be created)
**Purpose**: Clustering analysis
- Compare clustering algorithms
- Determine optimal k
- Apply best clustering method
- Validate results

**Runtime**: ~10-20 minutes

### 05_final_analysis.ipynb (To be created)
**Purpose**: Comprehensive analysis and visualization
- Cluster characterization
- Generate all visualizations
- Clinical interpretation
- Export final results

**Runtime**: ~15-30 minutes

## Usage Instructions

### 1. Start Jupyter

```bash
cd /app
jupyter notebook
```

### 2. Open Notebooks in Sequence

Work through notebooks 01 → 02 → 03 → 04 → 05 in order, as each depends on outputs from the previous one.

### 3. Run Cells

Execute cells sequentially using:
- **Shift + Enter**: Run cell and move to next
- **Ctrl + Enter**: Run cell and stay
- **Cell → Run All**: Run entire notebook

## Data Flow

```
01_data_exploration.ipynb
    ↓ (saves: notebook_psych_cohort_step1.csv)
02_feature_engineering.ipynb
    ↓ (saves: notebook_features_step2.csv)
03_embeddings.ipynb
    ↓ (saves: notebook_embeddings_step3.pkl)
04_clustering.ipynb
    ↓ (saves: notebook_clusters_step4.csv)
05_final_analysis.ipynb
    ↓ (generates final reports and visualizations)
```

## Tips

### Memory Management
- If running out of memory, restart kernel: `Kernel → Restart`
- Clear outputs: `Cell → All Output → Clear`
- Use `del` to free variables: `del large_dataframe`

### Saving Work
- Notebooks auto-save every 2 minutes
- Manual save: `Ctrl + S` or `File → Save`
- Export as Python: `File → Download as → Python (.py)`

### Debugging
- Use `print()` statements liberally
- Check variable types: `type(variable)`
- Inspect DataFrames: `df.head()`, `df.info()`, `df.describe()`

### Performance
- Use `%%time` magic to measure cell runtime
- Sample large datasets for testing: `df.sample(1000)`
- Monitor memory: `psych_df.memory_usage(deep=True).sum() / 1e6` (MB)

## Alternative: Run Full Pipeline

If you prefer to run the entire pipeline without notebooks:

```bash
python /app/main_pipeline.py
```

This will execute all steps automatically and save results to `/app/outputs/`.

## Troubleshooting

### Kernel Dies During Embedding Generation
- Reduce batch_size in config.yaml
- Use smaller dimensionality reduction target
- Consider running main_pipeline.py instead

### Import Errors
```python
# Add to first cell of each notebook
import sys
sys.path.append('/app')
```

### File Not Found
- Ensure previous notebook completed successfully
- Check that intermediate files were saved
- Verify paths in code match actual locations

## Output Locations

All intermediate outputs are saved to:
- `/app/outputs/notebook_*.csv` - Processed DataFrames
- `/app/outputs/notebook_*.pkl` - Pickled objects (embeddings, models)

Final pipeline outputs go to:
- `/app/outputs/embeddings/`
- `/app/outputs/clusters/`
- `/app/outputs/visualizations/`
- `/app/outputs/reports/`
