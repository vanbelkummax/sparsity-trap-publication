# Configs Directory

Configuration files for experiments, datasets, and models.

## Expected Structure

```
configs/
├── experiment/
│   └── factorial_cv.yaml       # Factorial design (4 configs × 3 folds)
├── data/
│   └── visium_hd_crc.yaml      # Visium HD CRC dataset config
└── model/
    ├── hist2st_decoder.yaml    # Hist2ST decoder architecture
    └── img2st_decoder.yaml     # Img2ST decoder architecture
```

## Configuration Format

YAML format with hierarchical structure:

### experiment/factorial_cv.yaml
```yaml
name: factorial_cv
folds: [1, 2, 3]
models:
  - decoder: hist2st
    loss: poisson
  - decoder: hist2st
    loss: mse
  - decoder: img2st
    loss: poisson
  - decoder: img2st
    loss: mse
```

### data/visium_hd_crc.yaml
```yaml
dataset: visium_hd_crc
resolution: 2um
patients: [P1, P2, P5]
genes: 50  # Top 50 by variance
preprocessing: raw_counts  # No normalization
```

### model/*.yaml
Architecture specifications, hyperparameters, training settings.

## Usage

Configs are loaded by training scripts:
```bash
python scripts/train_factorial.py --config configs/experiment/factorial_cv.yaml
```
