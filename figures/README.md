# Figures Directory

This directory contains all publication figures.

## Subdirectories

- **manuscript/** - Main figures for the paper (Figure 1-4, S1)
- **wsi_improved/** - Whole-slide image comparisons (WSI)
- **tiles/** - Tile-level (128×128 patch) comparisons

## Expected Files

### manuscript/
- `figure_1_combined.png` - Factorial design + per-gene analysis (4-panel)
- `figure_3_main_effects.png` - Main effects decomposition
- `figure_4_representative_genes.png` - Representative genes by category
- `figure_wsi_comparison_composite.png` - WSI comparisons (double-column)
- `figure_s1_fold_consistency.png` - Cross-validation consistency

### wsi_improved/
WSI comparisons for 19 top genes (e.g., CEACAM5, EPCAM, KRT8, etc.)

### tiles/
Tile-level comparisons for the same genes

## Generation

Run scripts to generate figures:
```bash
python scripts/create_manuscript_figures.py
python scripts/create_wsi_and_tiles.py
```
