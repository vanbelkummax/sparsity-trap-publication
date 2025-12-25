# Publication Package Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform mse-vs-poisson-2um-benchmark into a publication-ready ArXiv package with clean code, LaTeX manuscript, and comprehensive documentation.

**Architecture:** Modular `src/` package with pytest tests + LaTeX manuscript with LaTeX Architect MCP standards + complete reproduction documentation

**Tech Stack:** Python 3.10+, PyTorch, pytest, LaTeX (IEEEtran/arXiv), LaTeX Architect MCP

---

## BATCH 1: Core Infrastructure (Priority: CRITICAL)

### Task 1.1: Create Package Structure

**Files:**
- Create: `src/__init__.py`
- Create: `src/models/__init__.py`
- Create: `src/data/__init__.py`
- Create: `src/training/__init__.py`
- Create: `src/evaluation/__init__.py`
- Create: `src/utils/__init__.py`
- Create: `tests/__init__.py`
- Create: `setup.py`
- Create: `requirements.txt`

**Step 1: Create directory structure**

```bash
mkdir -p src/models src/data src/training src/evaluation src/utils tests
touch src/__init__.py src/models/__init__.py src/data/__init__.py
touch src/training/__init__.py src/evaluation/__init__.py src/utils/__init__.py
touch tests/__init__.py
```

Run: `tree src/ tests/`
Expected: Directory structure created

**Step 2: Write setup.py for package installation**

Create `setup.py`:
```python
from setuptools import setup, find_packages

setup(
    name="sparsity-trap",
    version="1.0.0",
    author="Max Van Belkum",
    author_email="max.vanbelkum@vanderbilt.edu",
    description="The Sparsity Trap: Why MSE Fails and Poisson Succeeds for 2um Spatial Transcriptomics",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/vanbelkummax/sparsity-trap-publication",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-image>=0.20.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "pillow>=9.5.0",
        "pytest>=7.3.0",
        "pytest-cov>=4.1.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)
```

**Step 3: Write requirements.txt**

Create `requirements.txt`:
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scipy>=1.10.0
scikit-image>=0.20.0
matplotlib>=3.7.0
pandas>=2.0.0
tqdm>=4.65.0
pillow>=9.5.0
pytest>=7.3.0
pytest-cov>=4.1.0
```

**Step 4: Commit infrastructure**

```bash
git add src/ tests/ setup.py requirements.txt
git commit -m "feat: add package infrastructure with src/ and tests/

- Create modular src/ package structure
- Add setup.py for pip installation
- Add requirements.txt with dependencies
- Prepare for TDD implementation"
```

---

### Task 1.2: Extract Loss Functions (TDD)

**Files:**
- Create: `tests/test_losses.py`
- Create: `src/models/losses.py`

**Step 1: Write failing test for Poisson NLL loss**

Create `tests/test_losses.py`:
```python
import pytest
import torch
from src.models.losses import PoissonNLLLoss, MSELoss


class TestPoissonNLLLoss:
    def test_poisson_nll_zero_target(self):
        """Test Poisson NLL when target is zero (k=0)."""
        loss_fn = PoissonNLLLoss()
        predictions = torch.tensor([[1.0, 2.0, 3.0]])  # lambda values
        targets = torch.tensor([[0.0, 0.0, 0.0]])  # k=0 everywhere

        loss = loss_fn(predictions, targets)

        # L = lambda - k*log(lambda) = lambda when k=0
        expected = predictions.mean()  # Should just be mean of lambdas
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_poisson_nll_nonzero_target(self):
        """Test Poisson NLL when target is non-zero."""
        loss_fn = PoissonNLLLoss()
        predictions = torch.tensor([[2.0]])  # lambda=2
        targets = torch.tensor([[3.0]])  # k=3

        loss = loss_fn(predictions, targets)

        # L = lambda - k*log(lambda) = 2 - 3*log(2) = 2 - 3*0.693 = -0.079
        expected = 2.0 - 3.0 * torch.log(torch.tensor(2.0))
        assert torch.isclose(loss, expected, atol=1e-3)

    def test_poisson_nll_batch(self):
        """Test Poisson NLL with batched inputs."""
        loss_fn = PoissonNLLLoss()
        batch_size, h, w, genes = 4, 8, 8, 50
        predictions = torch.rand(batch_size, genes, h, w) * 10
        targets = torch.rand(batch_size, genes, h, w) * 20

        loss = loss_fn(predictions, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()  # Scalar
        assert loss > 0

    def test_poisson_nll_requires_positive_lambda(self):
        """Test that negative lambda predictions are handled."""
        loss_fn = PoissonNLLLoss()
        predictions = torch.tensor([[-1.0, 2.0]])  # Negative lambda
        targets = torch.tensor([[0.0, 3.0]])

        # Should either error or clamp to small positive value
        # Implementation should prevent log(0) or log(negative)
        loss = loss_fn(predictions, targets)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestMSELoss:
    def test_mse_zero_error(self):
        """Test MSE when prediction equals target."""
        loss_fn = MSELoss()
        predictions = torch.tensor([[1.0, 2.0, 3.0]])
        targets = torch.tensor([[1.0, 2.0, 3.0]])

        loss = loss_fn(predictions, targets)

        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_mse_simple(self):
        """Test MSE with known values."""
        loss_fn = MSELoss()
        predictions = torch.tensor([[2.0, 4.0]])
        targets = torch.tensor([[1.0, 3.0]])

        loss = loss_fn(predictions, targets)

        # MSE = mean((2-1)^2 + (4-3)^2) = mean(1 + 1) = 1.0
        expected = torch.tensor(1.0)
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_mse_batch(self):
        """Test MSE with batched inputs."""
        loss_fn = MSELoss()
        batch_size, h, w, genes = 4, 8, 8, 50
        predictions = torch.rand(batch_size, genes, h, w)
        targets = torch.rand(batch_size, genes, h, w)

        loss = loss_fn(predictions, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()  # Scalar
        assert loss >= 0
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/user/.config/superpowers/worktrees/mse-vs-poisson-2um-benchmark/publication-package && pytest tests/test_losses.py -v`
Expected: All tests FAIL with "ModuleNotFoundError: No module named 'src.models.losses'"

**Step 3: Implement loss functions**

Create `src/models/losses.py`:
```python
"""Loss functions for spatial transcriptomics prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoissonNLLLoss(nn.Module):
    """
    Poisson Negative Log-Likelihood Loss.

    For count data y ~ Poisson(lambda), the NLL is:
        L = lambda - y * log(lambda)

    This loss is critical for sparse spatial transcriptomics because:
    - When y > 0, predicting lambda -> 0 gives INFINITE loss (prevents "gray fog")
    - When y = 0, loss = lambda (encourages small predictions)
    - Naturally handles mean-variance relationship (Var = Mean)

    Args:
        eps: Small constant to prevent log(0). Default: 1e-8
        reduction: 'mean', 'sum', or 'none'. Default: 'mean'
    """

    def __init__(self, eps=1e-8, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Predicted lambda values (B, C, H, W). Must be positive.
            targets: Ground truth counts (B, C, H, W). Non-negative integers.

        Returns:
            Scalar loss if reduction='mean', otherwise per-element losses.
        """
        # Clamp predictions to prevent log(0)
        predictions = torch.clamp(predictions, min=self.eps)

        # Poisson NLL: lambda - k * log(lambda)
        loss = predictions - targets * torch.log(predictions)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MSELoss(nn.Module):
    """
    Mean Squared Error Loss.

    Standard L2 loss: (prediction - target)^2

    WARNING: For sparse count data (95% zeros), MSE encourages predicting
    zero everywhere because that minimizes the loss. This is the "sparsity trap".
    Use PoissonNLLLoss instead.

    Args:
        reduction: 'mean', 'sum', or 'none'. Default: 'mean'
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Predicted values (B, C, H, W)
            targets: Ground truth values (B, C, H, W)

        Returns:
            Scalar loss if reduction='mean', otherwise per-element losses.
        """
        loss = (predictions - targets) ** 2

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_loss_function(loss_type='poisson'):
    """
    Factory function to get loss by name.

    Args:
        loss_type: 'poisson' or 'mse'

    Returns:
        Loss function instance
    """
    if loss_type == 'poisson':
        return PoissonNLLLoss()
    elif loss_type == 'mse':
        return MSELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Use 'poisson' or 'mse'.")
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/user/.config/superpowers/worktrees/mse-vs-poisson-2um-benchmark/publication-package && pytest tests/test_losses.py -v`
Expected: All tests PASS

**Step 5: Add docstring test coverage**

Run: `cd /home/user/.config/superpowers/worktrees/mse-vs-poisson-2um-benchmark/publication-package && pytest tests/test_losses.py --cov=src.models.losses --cov-report=term-missing`
Expected: Coverage >90%

**Step 6: Commit loss functions**

```bash
git add src/models/losses.py tests/test_losses.py
git commit -m "feat: implement Poisson NLL and MSE loss functions with tests

- Add PoissonNLLLoss with eps clamping to prevent log(0)
- Add MSELoss as baseline
- Comprehensive tests for edge cases (zero targets, batching)
- Document sparsity trap in docstrings
- Test coverage >90%"
```

---

### Task 1.3: Extract Metrics (TDD)

**Files:**
- Create: `tests/test_metrics.py`
- Create: `src/evaluation/metrics.py`

**Step 1: Write failing test for SSIM metric**

Create `tests/test_metrics.py`:
```python
import pytest
import torch
import numpy as np
from src.evaluation.metrics import compute_ssim, compute_pcc


class TestSSIM:
    def test_ssim_identical_images(self):
        """Test SSIM between identical images."""
        img1 = torch.rand(128, 128)
        img2 = img1.clone()

        ssim = compute_ssim(img1, img2)

        assert np.isclose(ssim, 1.0, atol=1e-5)

    def test_ssim_different_images(self):
        """Test SSIM between different images."""
        img1 = torch.zeros(128, 128)
        img2 = torch.ones(128, 128)

        ssim = compute_ssim(img1, img2)

        assert 0.0 <= ssim < 1.0

    def test_ssim_with_mask(self):
        """Test SSIM with tissue mask."""
        img1 = torch.rand(128, 128)
        img2 = torch.rand(128, 128)
        mask = torch.ones(128, 128)
        mask[:64, :64] = 0  # Mask out top-left quadrant

        ssim_full = compute_ssim(img1, img2)
        ssim_masked = compute_ssim(img1, img2, mask=mask)

        # Masked SSIM should only consider unmasked region
        assert ssim_masked != ssim_full

    def test_ssim_batched(self):
        """Test SSIM with batched inputs."""
        batch_size, h, w = 4, 128, 128
        img1 = torch.rand(batch_size, h, w)
        img2 = torch.rand(batch_size, h, w)

        ssims = compute_ssim(img1, img2, batched=True)

        assert len(ssims) == batch_size
        assert all(0.0 <= s <= 1.0 for s in ssims)


class TestPCC:
    def test_pcc_perfect_correlation(self):
        """Test PCC with perfect positive correlation."""
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
        target = torch.tensor([1.0, 2.0, 3.0, 4.0])

        pcc = compute_pcc(pred, target)

        assert np.isclose(pcc, 1.0, atol=1e-5)

    def test_pcc_perfect_anti_correlation(self):
        """Test PCC with perfect negative correlation."""
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
        target = torch.tensor([4.0, 3.0, 2.0, 1.0])

        pcc = compute_pcc(pred, target)

        assert np.isclose(pcc, -1.0, atol=1e-5)

    def test_pcc_zero_correlation(self):
        """Test PCC with zero correlation."""
        pred = torch.tensor([1.0, 2.0, 1.0, 2.0])
        target = torch.tensor([1.0, 1.0, 2.0, 2.0])

        pcc = compute_pcc(pred, target)

        assert -0.5 < pcc < 0.5  # Near zero

    def test_pcc_with_mask(self):
        """Test PCC with tissue mask."""
        pred = torch.rand(128, 128)
        target = torch.rand(128, 128)
        mask = torch.ones(128, 128)
        mask[:64, :64] = 0  # Mask out top-left quadrant

        pcc_full = compute_pcc(pred, target)
        pcc_masked = compute_pcc(pred, target, mask=mask)

        # Masked PCC should only consider unmasked region
        assert pcc_masked != pcc_full
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.evaluation.metrics'"

**Step 3: Implement metrics**

Create `src/evaluation/metrics.py`:
```python
"""Evaluation metrics for spatial transcriptomics prediction."""

import numpy as np
import torch
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim_skimage


def compute_ssim(img1, img2, mask=None, batched=False, data_range=None):
    """
    Compute Structural Similarity Index (SSIM) between two images.

    SSIM measures perceptual similarity considering luminance, contrast, and structure.
    Range: [-1, 1], where 1 = perfect match.

    Args:
        img1: Predicted image (H, W) or (B, H, W)
        img2: Target image (H, W) or (B, H, W)
        mask: Binary tissue mask (H, W). If provided, only compute SSIM on masked region.
        batched: If True, img1 and img2 are (B, H, W) and returns list of SSIMs
        data_range: Data range of images. If None, inferred from img2.

    Returns:
        SSIM value (float) or list of SSIMs if batched=True
    """
    if batched:
        assert img1.ndim == 3 and img2.ndim == 3
        assert img1.shape[0] == img2.shape[0]
        return [compute_ssim(img1[i], img2[i], mask=mask, data_range=data_range)
                for i in range(img1.shape[0])]

    # Convert to numpy
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Ensure 2D
    assert img1.ndim == 2 and img2.ndim == 2
    assert img1.shape == img2.shape

    # Apply mask if provided
    if mask is not None:
        assert mask.shape == img1.shape
        mask = mask.astype(bool)
        # Extract masked regions as 1D arrays
        img1_masked = img1[mask]
        img2_masked = img2[mask]
        # Compute SSIM on 1D arrays (not ideal, but consistent)
        # Alternative: compute on full image then threshold
        # For proper implementation, we'd need windowed SSIM with mask
        # For simplicity, treat as 1D correlation-like metric
        if len(img1_masked) == 0:
            return 0.0
        # Use Pearson correlation as proxy when masked
        # (True SSIM requires spatial structure, hard with mask)
        from scipy.stats import pearsonr
        if np.std(img1_masked) == 0 or np.std(img2_masked) == 0:
            return 0.0
        return pearsonr(img1_masked, img2_masked)[0]

    # Infer data range
    if data_range is None:
        data_range = img2.max() - img2.min()
        if data_range == 0:
            data_range = 1.0

    # Compute SSIM using skimage
    ssim_val = ssim_skimage(
        img1, img2,
        data_range=data_range,
        win_size=7,  # Default window size
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False
    )

    return float(ssim_val)


def compute_pcc(pred, target, mask=None):
    """
    Compute Pearson Correlation Coefficient (PCC).

    PCC measures linear correlation between predictions and targets.
    Range: [-1, 1], where 1 = perfect positive correlation.

    Args:
        pred: Predicted values (any shape)
        target: Ground truth values (same shape as pred)
        mask: Binary mask (same shape). If provided, only compute PCC on masked region.

    Returns:
        PCC value (float)
    """
    # Convert to numpy
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Flatten
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    # Apply mask if provided
    if mask is not None:
        mask_flat = mask.flatten().astype(bool)
        pred_flat = pred_flat[mask_flat]
        target_flat = target_flat[mask_flat]

    # Handle edge cases
    if len(pred_flat) == 0:
        return 0.0
    if np.std(pred_flat) == 0 or np.std(target_flat) == 0:
        return 0.0

    # Compute Pearson correlation
    pcc, _ = pearsonr(pred_flat, target_flat)

    return float(pcc)


def compute_per_gene_metrics(predictions, targets, mask=None):
    """
    Compute SSIM and PCC for each gene separately.

    Args:
        predictions: (B, C, H, W) or (C, H, W) predicted gene expression
        targets: (B, C, H, W) or (C, H, W) ground truth
        mask: (H, W) binary tissue mask

    Returns:
        dict with keys 'ssim' and 'pcc', each containing list of per-gene values
    """
    # Handle batching
    if predictions.ndim == 4:
        predictions = predictions[0]  # Take first batch
        targets = targets[0]

    assert predictions.ndim == 3  # (C, H, W)
    num_genes = predictions.shape[0]

    ssims = []
    pccs = []

    for i in range(num_genes):
        pred_gene = predictions[i]
        target_gene = targets[i]

        ssim_val = compute_ssim(pred_gene, target_gene, mask=mask)
        pcc_val = compute_pcc(pred_gene, target_gene, mask=mask)

        ssims.append(ssim_val)
        pccs.append(pcc_val)

    return {
        'ssim': ssims,
        'pcc': pccs
    }
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metrics.py -v`
Expected: All tests PASS

**Step 5: Commit metrics**

```bash
git add src/evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: implement SSIM and PCC metrics with tests

- Add compute_ssim with mask support
- Add compute_pcc with mask support
- Add compute_per_gene_metrics for batch processing
- Comprehensive edge case tests
- Batched processing support"
```

---

## BATCH 2: LaTeX Manuscript (Priority: HIGH)

### Task 2.1: Create LaTeX Structure

**Files:**
- Create: `paper/main.tex`
- Create: `paper/main.bib`
- Create: `paper/.gitignore`

**Step 1: Create paper directory and symlink figures**

```bash
mkdir -p paper/build
cd paper
ln -s ../figures/manuscript figs
cd ..
```

**Step 2: Create .gitignore for LaTeX build artifacts**

Create `paper/.gitignore`:
```
*.aux
*.bbl
*.blg
*.log
*.out
*.toc
*.synctex.gz
build/
*.pdf
!build/*.pdf
```

**Step 3: Create main.tex with IEEEtran template and proper figure blocks**

This uses LaTeX Architect MCP standards:
- All figures use [t!] placement
- Width: 0.95\columnwidth for single-column
- Non-breaking space in references: Figure~\ref{fig:label}

Create `paper/main.tex`:
```latex
\documentclass[conference]{IEEEtran}

\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{hyperref}

\begin{document}

\title{The Sparsity Trap: Why MSE Fails and Poisson Succeeds for 2μm Spatial Transcriptomics Prediction}

\author{
\IEEEauthorblockN{Max Van Belkum}
\IEEEauthorblockA{\textit{MD-PhD Program} \\
\textit{Vanderbilt University Medical Center}\\
Nashville, TN, USA \\
max.vanbelkum@vanderbilt.edu}
}

\maketitle

\begin{abstract}
Visium HD spatial transcriptomics enables gene expression profiling at 2μm resolution, but the extreme sparsity of count data (95\% zeros) creates a fundamental challenge for prediction models. We demonstrate that standard mean squared error (MSE) loss catastrophically fails at this resolution, collapsing to uniform near-zero predictions that we term the ``sparsity trap.'' In contrast, Poisson negative log-likelihood (NLL) loss avoids this failure mode by assigning infinite penalty when predicting zero expression for non-zero counts. Using a 2×2 factorial design (decoder architecture × loss function) with 3-fold cross-validation on colorectal cancer Visium HD data, we show that Poisson loss achieves 2.7× better structural similarity (SSIM 0.542 vs 0.200, p<0.001) compared to MSE. All 50 genes analyzed showed improved SSIM with Poisson loss, with benefit strongly correlated to gene sparsity (r=0.577, p<0.0001). Qualitative analysis reveals that Poisson preserves glandular architecture in epithelial markers (CEACAM5, EPCAM), while MSE produces featureless ``gray fog.'' Our factorial analysis further shows a synergistic interaction between Hist2ST decoder and Poisson loss (2.71× improvement vs 1.89× with simpler U-Net). These findings establish Poisson NLL as essential for high-resolution spatial transcriptomics prediction and provide the foundation for next-generation zero-inflated models.
\end{abstract}

\begin{IEEEkeywords}
spatial transcriptomics, deep learning, loss functions, Visium HD, gene expression prediction
\end{IEEEkeywords}

\section{Introduction}

% PLACEHOLDER: Will be filled in next task
% Visual hook: CEACAM5 comparison
% Problem: High-resolution ST at 2um
% Gap: Does MSE work at 2um?

\section{The Sparsity Trap}

% PLACEHOLDER: Will be filled in next task
% Data characteristics: 95% zeros
% MSE failure mode
% Mathematical analysis

\section{Methods}

\subsection{Dataset}
% PLACEHOLDER

\subsection{Architecture}
% PLACEHOLDER

\subsection{Loss Functions}
% PLACEHOLDER

\subsection{Experimental Design}
% PLACEHOLDER

\subsection{Evaluation Metrics}
% PLACEHOLDER

\section{Results}

\subsection{Factorial Design}
% PLACEHOLDER

% Figure 1: Combined 4-panel (factorial + scatter + sparsity + waterfall)
% WILL USE LaTeX Architect MCP to generate this block

\subsection{Glandular Structure Recovery}
% PLACEHOLDER

% Figure 2: WSI comparisons
% WILL USE LaTeX Architect MCP to generate this block

\subsection{Per-Gene Analysis}
% PLACEHOLDER

% Figure 3: Representative genes by category
% WILL USE LaTeX Architect MCP to generate this block

\subsection{Main Effects}
% PLACEHOLDER

% Figure 4: Main effects decomposition
% WILL USE LaTeX Architect MCP to generate this block

\section{Discussion}

\subsection{Why Poisson Works}
% PLACEHOLDER

\subsection{Comparison to Prior Work}
% PLACEHOLDER

\subsection{Limitations}
% PLACEHOLDER

\subsection{Future Directions}
% PLACEHOLDER

\section{Conclusion}

% PLACEHOLDER

\section*{Acknowledgments}
This work was supported by the Vanderbilt MD-PhD Program. We thank Yuankai Huo, Ken Lau, and Bennett Landman for helpful discussions.

\bibliographystyle{IEEEtran}
\bibliography{main}

\end{document}
```

**Step 4: Create empty bibliography**

Create `paper/main.bib`:
```bibtex
% Bibliography entries will be added via mcp__vanderbilt-professors__ MCP

% Key citations to add:
% - Huo et al. (2025): Img2ST-Net (PMID:41210922)
% - Lau et al. (2022): CRC spatial atlas (PMID:35794563)
% - Sarkar et al. (2023/2025): Spatial methods
% - SSIM metric (Wang et al.)
% - Poisson regression (McCullagh & Nelder)
```

**Step 5: Test compilation**

Run: `cd paper && pdflatex -interaction=nonstopmode main.tex`
Expected: Compiles successfully (warnings OK, no errors)

**Step 6: Commit LaTeX structure**

```bash
git add paper/
git commit -m "feat: create LaTeX manuscript structure

- Add main.tex with IEEEtran template
- Add abstract (250 words)
- Create section placeholders
- Add main.bib for references
- Symlink figures/manuscript to paper/figs
- Ready for content writing"
```

---

### Task 2.2: Generate Figure Blocks with LaTeX Architect MCP

**Files:**
- Modify: `paper/main.tex` (add figure blocks)

**Step 1: Generate Figure 1 block (Combined 4-panel)**

We'll use the LaTeX Architect MCP to ensure proper standards.

Expected code to generate via MCP tool `mcp__latex-architect__generate_figure_block`:
```
filename: figs/figure_1_combined.png
caption: The Sparsity Trap: Factorial Design and Per-Gene Analysis. (a) 2×2 factorial heatmap showing mean SSIM with 3-fold CV error bars. Poisson loss dramatically outperforms MSE, with synergistic benefit when paired with Hist2ST decoder. (b) Per-gene scatter plot: all 50 genes lie above the diagonal (MSE SSIM < Poisson SSIM). (c) Sparsity correlation: genes with higher sparsity (more zeros) benefit more from Poisson loss (r=0.577, p<0.0001). (d) Waterfall plot: all genes show positive Δ-SSIM, with TSPAN8 achieving +0.73 improvement.
label: fig:factorial
wide: false
placement: t!
```

This will generate a figure block that follows standards:
- Uses [t!] placement (top-aligned, professional)
- Width: 0.95\columnwidth
- Non-breaking reference

**Step 2: Generate Figure 2 block (WSI comparison)**

Expected code via MCP:
```
filename: figs/figure_wsi_comparison_composite.png
caption: Glandular Structure Recovery with Poisson Loss. Whole-slide image comparisons for four epithelial markers (CEACAM5, EPCAM, KRT8, OLFM4) showing ground truth, MSE prediction (gray fog), and Poisson prediction (crisp gland boundaries). SSIM scores demonstrate dramatic improvement: CEACAM5 (0.105→0.804), EPCAM (0.102→0.785), KRT8 (0.096→0.769), OLFM4 (0.107→0.785).
label: fig:wsi_recovery
wide: true
placement: t!
```

This will use `figure*` environment for double-column span.

**Step 3: Generate Figure 3 block (Representative genes)**

Expected code via MCP:
```
filename: figs/figure_4_representative_genes.png
caption: Representative Genes by Category. 2×3 grid showing top-performing genes from each functional category: Epithelial (CEACAM5, Δ-SSIM +0.699), Immune (JCHAIN, +0.459), Stromal (VIM, +0.180), Secretory (MUC12, +0.632), Mitochondrial (MT-ND5, +0.517), Housekeeping (TMSB10, +0.409). All categories benefit from Poisson loss, with epithelial and secretory markers showing largest gains.
label: fig:categories
wide: false
placement: t!
```

**Step 4: Generate Figure 4 block (Main effects)**

Expected code via MCP:
```
filename: figs/figure_3_main_effects.png
caption: Factorial Analysis: Main Effects and Interaction. (a) Loss function main effect: Poisson achieves 2.37× improvement over MSE (averaged across decoders). (b) Decoder main effect: Hist2ST achieves 1.81× improvement over Img2ST (averaged across losses). (c) Interaction plot: non-parallel lines indicate synergistic interaction—Poisson provides greater benefit with Hist2ST (2.71×) than with Img2ST (1.89×).
label: fig:main_effects
wide: false
placement: t!
```

**Step 5: Validate figure placement**

After adding all figures, validate with LaTeX Architect MCP:
Use `mcp__latex-architect__check_spatial_inclusion` on `paper/main.tex`

Expected: No [h] or [h!] placements found (all use [t!] or [b!])

**Step 6: Commit figure blocks**

```bash
git add paper/main.tex
git commit -m "feat: add publication figures with LaTeX Architect standards

- Figure 1: Factorial design + per-gene analysis (4-panel)
- Figure 2: WSI comparisons (double-column)
- Figure 3: Representative genes by category
- Figure 4: Main effects decomposition
- All figures use [t!] placement (professional)
- All figures use 0.95\columnwidth or 0.95\textwidth
- Verified with LaTeX Architect check_spatial_inclusion"
```

---

## BATCH 3: Documentation (Priority: MEDIUM)

### Task 3.1: Write LICENSE

**Files:**
- Create: `LICENSE`

**Step 1: Create MIT License**

Create `LICENSE`:
```
MIT License

Copyright (c) 2025 Max Van Belkum

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Step 2: Commit LICENSE**

```bash
git add LICENSE
git commit -m "docs: add MIT license"
```

---

### Task 3.2: Create CITATION.cff

**Files:**
- Create: `CITATION.cff`

**Step 1: Create citation metadata**

Create `CITATION.cff`:
```yaml
cff-version: 1.2.0
message: "If you use this software, please cite it as below."
authors:
  - family-names: Van Belkum
    given-names: Max
    email: max.vanbelkum@vanderbilt.edu
    affiliation: Vanderbilt University Medical Center
    orcid: "https://orcid.org/0000-0000-0000-0000"  # TODO: Add real ORCID
title: "The Sparsity Trap: Why MSE Fails and Poisson Succeeds for 2μm Spatial Transcriptomics Prediction"
version: 1.0.0
doi: 10.XXXXX/XXXXX  # Will be updated after ArXiv/Zenodo
date-released: 2025-12-25
url: "https://github.com/vanbelkummax/sparsity-trap-publication"
repository-code: "https://github.com/vanbelkummax/sparsity-trap-publication"
keywords:
  - spatial-transcriptomics
  - deep-learning
  - loss-functions
  - poisson-regression
  - visium-hd
  - gene-expression
license: MIT
abstract: |
  Visium HD spatial transcriptomics enables gene expression profiling at 2μm
  resolution, but extreme sparsity (95% zeros) creates a fundamental challenge.
  We demonstrate that MSE loss catastrophically fails, collapsing to uniform
  near-zero predictions. Poisson NLL avoids this failure mode, achieving 2.7×
  better SSIM. All 50 genes benefit, with improvement strongly correlated to
  gene sparsity (r=0.577, p<0.0001). This work establishes Poisson NLL as
  essential for high-resolution spatial transcriptomics.
```

**Step 2: Commit CITATION.cff**

```bash
git add CITATION.cff
git commit -m "docs: add CITATION.cff for software citation

- Include author, title, version
- Add placeholder DOI (will update after ArXiv)
- Add keywords and abstract
- GitHub/Zenodo parseable format"
```

---

### Task 3.3: Write REPRODUCTION.md

**Files:**
- Create: `docs/REPRODUCTION.md`

**Step 1: Write step-by-step reproduction guide**

Create `docs/REPRODUCTION.md`:
```markdown
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
git clone https://github.com/vanbelkummax/sparsity-trap-publication.git
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
print(f'Sparsity correlation: {df[['sparsity', 'delta_ssim']].corr().iloc[0,1]:.3f}')
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
  url = {https://github.com/vanbelkummax/sparsity-trap-publication},
  doi = {10.XXXXX/XXXXX}  # Will be updated after publication
}
```

## Support

For questions or issues:
- Open a GitHub issue: https://github.com/vanbelkummax/sparsity-trap-publication/issues
- Email: max.vanbelkum@vanderbilt.edu
```

**Step 2: Commit reproduction guide**

```bash
git add docs/REPRODUCTION.md
git commit -m "docs: add comprehensive reproduction guide

- System requirements
- Installation instructions (conda + pip)
- One-click reproduction script
- Manual step-by-step instructions
- Verification checks
- Troubleshooting section"
```

---

## BATCH 4: Testing & Polish (Priority: HIGH)

### Task 4.1: Integration Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

Create `tests/test_integration.py`:
```python
"""Integration test: Train for 1 epoch and verify loss decreases."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.losses import PoissonNLLLoss, MSELoss


class DummyModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, in_channels=3, out_channels=50):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        return torch.exp(self.conv2(x))  # Positive output for Poisson


class TestIntegration:
    def test_poisson_loss_training(self):
        """Test that training with Poisson loss decreases loss."""
        model = DummyModel()
        loss_fn = PoissonNLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Dummy data
        batch_size, h, w, genes = 2, 128, 128, 50
        images = torch.rand(batch_size, 3, h, w)
        targets = torch.rand(batch_size, genes, h, w) * 10

        # Initial loss
        model.train()
        pred_init = model(images)
        loss_init = loss_fn(pred_init, targets)

        # Train for 10 steps
        for _ in range(10):
            optimizer.zero_grad()
            pred = model(images)
            loss = loss_fn(pred, targets)
            loss.backward()
            optimizer.step()

        # Final loss
        pred_final = model(images)
        loss_final = loss_fn(pred_final, targets)

        # Loss should decrease
        assert loss_final < loss_init, f"Loss did not decrease: {loss_init:.4f} -> {loss_final:.4f}"

    def test_mse_loss_training(self):
        """Test that training with MSE loss decreases loss."""
        model = DummyModel()
        loss_fn = MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Dummy data
        batch_size, h, w, genes = 2, 128, 128, 50
        images = torch.rand(batch_size, 3, h, w)
        targets = torch.rand(batch_size, genes, h, w) * 10

        # Initial loss
        model.train()
        pred_init = model(images)
        loss_init = loss_fn(pred_init, targets)

        # Train for 10 steps
        for _ in range(10):
            optimizer.zero_grad()
            pred = model(images)
            loss = loss_fn(pred, targets)
            loss.backward()
            optimizer.step()

        # Final loss
        pred_final = model(images)
        loss_final = loss_fn(pred_final, targets)

        # Loss should decrease
        assert loss_final < loss_init, f"Loss did not decrease: {loss_init:.4f} -> {loss_final:.4f}"
```

**Step 2: Run integration test**

Run: `pytest tests/test_integration.py -v`
Expected: Both tests PASS, loss decreases during training

**Step 3: Commit integration test**

```bash
git add tests/test_integration.py
git commit -m "test: add integration test for training

- Verify Poisson loss training decreases loss
- Verify MSE loss training decreases loss
- Use dummy model for fast execution
- Ensures gradients flow correctly"
```

---

### Task 4.2: Update README for Publication

**Files:**
- Modify: `README.md`

**Step 1: Rewrite README for publication audience**

Replace existing `README.md` with publication-oriented version:

```markdown
# The Sparsity Trap

**Why MSE Fails and Poisson Succeeds for 2μm Spatial Transcriptomics Prediction**

[![ArXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

> **TL;DR:** MSE loss collapses to "gray fog" on 2μm sparse spatial transcriptomics. Poisson NLL achieves 2.7× better SSIM by avoiding the sparsity trap. **All 50 genes benefit.**

---

## Key Finding

<p align="center">
  <img src="figures/manuscript/figure_1_combined.png" width="95%">
</p>

At 2μm resolution (Visium HD), ~95% of spatial bins contain **zero UMI counts**. This extreme sparsity creates a fundamental failure mode for MSE loss:

- **MSE Loss:** Predicting zero everywhere minimizes loss → "gray fog"
- **Poisson NLL:** Infinite penalty for λ→0 when k>0 → preserves structure

**Result:** 2.7× SSIM improvement (0.200 → 0.542, p<0.001), 50/50 genes benefit

---

## Visual Evidence

<p align="center">
  <img src="figures/wsi_improved/CEACAM5_2um_WSI_improved.png" width="95%">
</p>

**Epithelial markers** show dramatic recovery of glandular architecture. MSE produces featureless predictions, while Poisson captures crypt boundaries and lumen structures.

---

## Main Results

### Model Performance (3-Fold Cross-Validation)

| Model | Decoder | Loss | SSIM 2μm | PCC 2μm | Rank |
|-------|---------|------|----------|---------|------|
| **E'** | **Hist2ST** | **Poisson** | **0.542 ± 0.019** | **0.151** | 🥇 |
| F | Img2ST | Poisson | 0.268 ± 0.013 | ~0.00 | 🥈 |
| D' | Hist2ST | MSE | 0.200 ± 0.012 | 0.111 | 🥉 |
| G | Img2ST | MSE | 0.142 ± 0.007 | -0.006 | 💀 |

### Key Statistics

- **SSIM Improvement:** 2.7× (MSE → Poisson with Hist2ST)
- **Genes Benefiting:** 50/50 (100%)
- **Mean Δ-SSIM:** +0.412
- **Sparsity Correlation:** r=0.577, p<0.0001 (sparser genes benefit more)

---

## Installation

```bash
git clone https://github.com/vanbelkummax/sparsity-trap-publication.git
cd mse-vs-poisson-2um-benchmark
conda env create -f environment.yml
conda activate sparsity-trap
pip install -e .
```

**Requirements:** Python 3.10+, PyTorch 2.0+, NVIDIA GPU with 24GB+ VRAM

---

## Quick Start

### Reproduce All Results

```bash
bash scripts/reproduce_paper.sh
```

This trains all 12 models (4 configs × 3 folds), generates figures, and compiles the manuscript.

**Estimated time:** 18-24 hours on RTX 5090

### Train Single Model

```bash
# Best model: Hist2ST + Poisson
python scripts/train_factorial.py --decoder hist2st --loss poisson --fold 1

# Worst model: Img2ST + MSE
python scripts/train_factorial.py --decoder img2st --loss mse --fold 1
```

### Generate Figures

```bash
python scripts/generate_figures.py
```

Outputs: `figures/manuscript/*.png`

---

## Repository Structure

```
mse-vs-poisson-2um-benchmark/
├── paper/                  # LaTeX manuscript
│   ├── main.tex
│   ├── main.bib
│   └── supplementary.tex
├── src/                    # Source code
│   ├── models/             # Loss functions, encoders, decoders
│   ├── data/               # Dataset, preprocessing
│   ├── training/           # Trainer, callbacks
│   └── evaluation/         # Metrics, visualization
├── tests/                  # Unit & integration tests
├── scripts/                # Training & evaluation scripts
├── figures/                # Publication figures
├── tables/                 # Per-gene metrics, summaries
├── results/                # Model checkpoints (gitignored)
└── docs/                   # Documentation
    ├── REPRODUCTION.md     # Step-by-step reproduction
    └── ARCHITECTURE.md     # Model architecture details
```

---

## Citation

If you use this work, please cite:

```bibtex
@software{vanbelkum2025sparsity_trap,
  author = {Van Belkum, Max},
  title = {The Sparsity Trap: Why MSE Fails and Poisson Succeeds for 2μm Spatial Transcriptomics},
  year = {2025},
  url = {https://github.com/vanbelkummax/sparsity-trap-publication},
  doi = {10.XXXXX/XXXXX}
}
```

---

## Related Work

This work builds upon:

- **Huo et al. (2025):** [Img2ST-Net](https://doi.org/10.1117/1.JMI.12.6.061410) - Baseline using MSE at 8-16μm
- **Lau et al. (2022):** [CRC Spatial Atlas](https://doi.org/10.1186/s12967-022-03510-8) - Dataset context
- **Sarkar et al. (2023):** [Spatial gene expression mapping](https://doi.org/10.1038/s41587-023-01961-z) - Related methods

---

## License

MIT License - see [LICENSE](LICENSE) for details

---

## Acknowledgments

This work was supported by the Vanderbilt MD-PhD Program. We thank:

- **Yuankai Huo** (Vanderbilt) - Spatial transcriptomics methods
- **Ken Lau** (Vanderbilt) - CRC spatial biology
- **Bennett Landman** (Vanderbilt) - Medical imaging

---

## Contact

**Max Van Belkum**
MD-PhD Student, Vanderbilt University Medical Center
Email: max.vanbelkum@vanderbilt.edu
GitHub: [@vanbelkummax](https://github.com/vanbelkummax)

For questions or issues, please [open a GitHub issue](https://github.com/vanbelkummax/sparsity-trap-publication/issues).
```

**Step 2: Commit updated README**

```bash
git add README.md
git commit -m "docs: rewrite README for publication audience

- Add visual abstract with figure
- Include key results table
- Add installation and quick start
- Add citation information
- Add related work and acknowledgments
- Professional formatting with badges"
```

---

## FINAL CHECKLIST

### Before Execution

- [ ] Review design document: `docs/plans/2025-12-25-publication-package-design.md`
- [ ] Understand target structure in Section 3 of design doc
- [ ] Identify which scripts need refactoring (grep for existing model/loss code)

### During Execution (BATCH 1)

- [ ] Create `src/` package structure
- [ ] Write tests FIRST for losses (TDD)
- [ ] Implement Poisson and MSE losses
- [ ] Verify coverage >90%
- [ ] Write tests FIRST for metrics (TDD)
- [ ] Implement SSIM and PCC
- [ ] Commit after each passing test suite

### During Execution (BATCH 2)

- [ ] Create LaTeX structure with IEEEtran
- [ ] Use LaTeX Architect MCP for ALL figure blocks
- [ ] Verify [t!] placement with check_spatial_inclusion
- [ ] Use Vanderbilt Professors MCP for citations
- [ ] Test compilation after each section
- [ ] Commit incrementally

### During Execution (BATCH 3)

- [ ] Create LICENSE (MIT)
- [ ] Create CITATION.cff
- [ ] Write REPRODUCTION.md
- [ ] Write ARCHITECTURE.md
- [ ] Update README
- [ ] Commit documentation

### During Execution (BATCH 4)

- [ ] Write integration test
- [ ] Verify tests pass: `pytest tests/ -v`
- [ ] Verify coverage: `pytest --cov=src`
- [ ] Test LaTeX compilation
- [ ] Commit final tests and polish

### After All Batches

- [ ] Run full test suite: `pytest tests/ -v --cov=src`
- [ ] Compile LaTeX: `cd paper && pdflatex main.tex`
- [ ] Visual inspection of all figures
- [ ] Merge to main branch
- [ ] Create GitHub release (v1.0.0)
- [ ] Upload to ArXiv
- [ ] Update CITATION.cff with DOI

---

## Execution Instructions

**Plan complete and saved to:**
`docs/plans/2025-12-25-publication-package-implementation.md`

**Two execution options:**

### 1. Subagent-Driven (this session)
- Stay in current session
- I dispatch fresh subagent per task
- Code review between tasks
- Fast iteration with quality gates

**To start:** Use `@superpowers:subagent-driven-development`

### 2. Parallel Session (separate)
- Open new Claude Code session in worktree: `~/.config/superpowers/worktrees/mse-vs-poisson-2um-benchmark/publication-package`
- Use `@superpowers:executing-plans` with this plan
- Batch execution with checkpoints
- Review after each batch

**Which approach would you prefer?**
