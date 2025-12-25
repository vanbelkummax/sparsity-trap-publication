# Reproduction Guide

This document provides step-by-step instructions to reproduce all results in "The Sparsity Trap" paper.

## System Requirements

- **OS:** Linux (tested on Ubuntu 22.04)
- **GPU:** NVIDIA GPU with 24GB+ VRAM (RTX 5090 or A100)
- **RAM:** 64GB+ recommended
- **Storage:** 200GB+ for data and results
- **Python:** 3.10 or later

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/vanbelkummax/mse-vs-poisson-2um-benchmark.git
cd mse-vs-poisson-2um-benchmark
```

### 2. Create Environment

**Option A: Conda (recommended)**
```bash
conda env create -f environment.yml
conda activate sparsity-trap
```

**Option B: pip**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .  # Install package in development mode
```

### 3. Verify Installation

```bash
pytest tests/ -v
```

Expected: All tests pass

## Data Download

The 10x Genomics Visium HD CRC dataset is publicly available:

```bash
# Download from 10x Genomics website
# https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-human-crc

# Or use provided download script
bash scripts/download_data.sh
```

Expected data structure:
```
data/
├── P1/
│   ├── raw_feature_bc_matrix.h5
│   ├── spatial/
│   └── ...
├── P2/
└── P5/
```

## Reproducing Results

### One-Click Reproduction (Recommended)

```bash
bash scripts/reproduce_paper.sh
```

This will:
1. Train all 12 models (4 configs × 3 folds)
2. Evaluate on test sets
3. Generate all figures
4. Compile LaTeX manuscript

Estimated time: 18-24 hours on RTX 5090

### Manual Step-by-Step

#### 1. Train Models (3-Fold CV)

Train all 4 configurations:

```bash
# E' (Hist2ST + Poisson) - Best model
python scripts/train_factorial.py --decoder hist2st --loss poisson --fold all

# D' (Hist2ST + MSE)
python scripts/train_factorial.py --decoder hist2st --loss mse --fold all

# F (Img2ST + Poisson)
python scripts/train_factorial.py --decoder img2st --loss poisson --fold all

# G (Img2ST + MSE) - Worst model
python scripts/train_factorial.py --decoder img2st --loss mse --fold all
```

Expected output: Model checkpoints in `results_cv/<model>/<fold>/`

#### 2. Evaluate Models

```bash
python scripts/evaluate_model.py --fold all
```

Expected output: Metrics JSON files with SSIM, PCC values

#### 3. Generate Figures

```bash
python scripts/generate_figures.py
```

Expected output:
- `figures/manuscript/figure_1_combined.png`
- `figures/manuscript/figure_3_main_effects.png`
- `figures/manuscript/figure_4_representative_genes.png`
- `figures/manuscript/figure_s1_fold_consistency.png`
- All WSI and tile comparisons

#### 4. Compile Manuscript

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
cd ..
```

Expected output: `paper/build/main.pdf`

## Verifying Results

### Key Metrics to Check

Open `results_cv/summary.json` and verify:

```json
{
  "E_prime": {
    "mean_ssim": 0.542,
    "std_ssim": 0.019,
    "mean_pcc_2um": 0.151
  },
  "D_prime": {
    "mean_ssim": 0.200,
    "std_ssim": 0.012,
    "mean_pcc_2um": 0.111
  }
}
```

**Critical check:** E' SSIM should be 2.7× better than D' (0.542 / 0.200 = 2.71)

### Per-Gene Analysis

```bash
python -c "
import pandas as pd
df = pd.read_csv('tables/table_s1_pergene_metrics.csv')
print(f'Genes with improved SSIM: {(df.delta_ssim > 0).sum()} / {len(df)}')
print(f'Mean delta-SSIM: {df.delta_ssim.mean():.3f}')
print(f'Sparsity correlation: {df[[\"sparsity\", \"delta_ssim\"]].corr().iloc[0,1]:.3f}')
"
```

Expected output:
```
Genes with improved SSIM: 50 / 50
Mean delta-SSIM: 0.412
Sparsity correlation: 0.577
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in config:
```bash
# Edit configs/experiment/factorial_cv.yaml
batch_size: 4  # Reduce from 8
```

### Missing Data

Ensure Visium HD data is downloaded to `data/` directory. Check paths in `configs/data/visium_hd_crc.yaml`.

### Figure Rendering Issues

Regenerate specific figure:
```bash
python scripts/generate_figures.py --figure figure_1_combined
```

## Citation

If you reproduce these results, please cite:

```bibtex
@software{vanbelkum2025sparsity_trap,
  author = {Van Belkum, Max},
  title = {The Sparsity Trap: Why MSE Fails and Poisson Succeeds for 2μm Spatial Transcriptomics},
  year = {2025},
  url = {https://github.com/vanbelkummax/mse-vs-poisson-2um-benchmark},
  doi = {10.XXXXX/XXXXX}  # Will be updated after publication
}
```

## Support

For questions or issues:
- Open a GitHub issue: https://github.com/vanbelkummax/mse-vs-poisson-2um-benchmark/issues
- Email: max.vanbelkum@vanderbilt.edu
