# Publication Package Design: The Sparsity Trap

**Date:** 2025-12-25
**Status:** Design Complete - Ready for Implementation
**Target:** ArXiv preprint (adaptable to MICCAI/Bioinformatics)

---

## Executive Summary

Transform the `mse-vs-poisson-2um-benchmark` repository into a publication-ready package documenting a critical methodological finding: **MSE loss catastrophically fails for 2μm sparse spatial transcriptomics, while Poisson NLL succeeds**. This is a direct advancement over recent work (Huo et al., 2025, PMID:41210922) and establishes the foundation for future ZINB-based improvements.

**Key Finding:** Poisson loss achieves 2.7× better SSIM than MSE (0.542 vs 0.200, p<0.001) on 2μm Visium HD data, with 100% of genes showing improvement.

---

## 1. Narrative Design

### Title
**"The Sparsity Trap: Why MSE Fails and Poisson Succeeds for 2μm Spatial Transcriptomics Prediction"**

### Abstract Structure (250 words)
1. **Motivation:** Visium HD enables 2μm resolution ST, but extreme sparsity (95% zeros) challenges prediction models
2. **Problem:** MSE loss collapses to "gray fog" - predicting near-zero everywhere
3. **Solution:** Poisson NLL with proper initialization avoids the sparsity trap
4. **Validation:** 2×2 factorial (decoder × loss) with 3-fold cross-validation on 3 CRC patients
5. **Results:** 2.7× SSIM improvement (0.200 → 0.542), 50/50 genes benefit, glandular structure recovered
6. **Impact:** Enables high-resolution spatial gene expression prediction at 2μm scale

### Main Sections
1. **Introduction**
   - Visual hook: CEACAM5 gland comparison (MSE fog vs Poisson crisp boundaries)
   - Problem statement: High-resolution ST prediction at 2μm
   - Current state: Existing methods use MSE at 8-16μm (Huo 2025, Sarkar 2023)
   - Gap: Does MSE work at 2μm? (Spoiler: No)

2. **The Sparsity Trap**
   - Data characteristics: 95% zero bins at 2μm
   - MSE failure mode: Predicting zero everywhere minimizes loss
   - Mathematical analysis: L2 vs Poisson NLL gradient behavior
   - Visualization: Loss surface comparison

3. **Methods**
   - Architecture: Virchow2 encoder + Hist2ST decoder
   - Loss functions: MSE vs Poisson NLL
   - Initialization: Bias = -3.0 for Poisson (critical detail)
   - Experimental design: 2×2 factorial, 3-fold CV
   - Metrics: SSIM (primary), PCC, per-gene analysis

4. **Results**
   - **Main Effect 1:** Loss function (Poisson 2.37× better than MSE)
   - **Main Effect 2:** Decoder architecture (Hist2ST 1.81× better than Img2ST)
   - **Interaction:** Synergistic benefit (2.71× when combined)
   - **Per-gene analysis:** 50/50 genes improve, sparsity correlation (r=0.577)
   - **Qualitative:** Glandular structure recovery (epithelial markers)

5. **Discussion**
   - Why Poisson works: Infinite penalty for λ→0 when k>0
   - Comparison to Huo et al. (2025): MSE at 8-16μm vs Poisson at 2μm
   - Limitations: 3 patients, CRC only, simple Poisson (future: ZINB)
   - Future directions: ZINB loss, cancer-optimized encoders (H-optimus-0)

6. **Conclusion**
   - Poisson loss is essential for 2μm ST prediction
   - Factorial validation establishes both factors matter
   - Enables next generation of high-resolution ST models

---

## 2. Figure Strategy

### Main Text Figures (5 total)

**Figure 1: The Sparsity Trap (4-panel composite)**
- **(a)** Factorial design: 2×2 heatmap (Decoder × Loss) with 3-fold CV error bars
- **(b)** Per-gene scatter: MSE SSIM vs Poisson SSIM (50/50 above diagonal)
- **(c)** Sparsity correlation: Gene sparsity vs Δ-SSIM (r=0.577, p<0.0001)
- **(d)** Waterfall plot: Genes ranked by Δ-SSIM (all positive)
- **File:** `figures/manuscript/figure_1_combined.png` (ALREADY EXISTS)
- **LaTeX generation:** Use `mcp__latex-architect__generate_figure_block`

**Figure 2: Glandular Structure Recovery (WSI comparison)**
- **Rows:** CEACAM5, EPCAM, KRT8, OLFM4 (epithelial markers)
- **Columns:** Ground Truth | MSE Prediction | Poisson Prediction
- **Annotations:** SSIM scores, gland boundary highlights
- **File:** Composite from `figures/wsi_improved/*.png`
- **LaTeX generation:** Use `mcp__latex-architect__generate_figure_block` with `wide=True`

**Figure 3: Representative Genes by Category**
- **Categories:** Epithelial, Immune, Stromal, Secretory, Mitochondrial, Housekeeping
- **Layout:** 2×3 grid of WSI comparisons
- **File:** `figures/manuscript/figure_4_representative_genes.png` (ALREADY EXISTS)
- **LaTeX generation:** Use `mcp__latex-architect__generate_figure_block`

**Figure 4: Main Effects Decomposition**
- **Panel A:** Loss effect (Poisson vs MSE, averaged across decoders)
- **Panel B:** Decoder effect (Hist2ST vs Img2ST, averaged across losses)
- **Panel C:** Interaction plot (non-parallel lines = synergy)
- **File:** `figures/manuscript/figure_3_main_effects.png` (ALREADY EXISTS)
- **LaTeX generation:** Use `mcp__latex-architect__generate_figure_block`

**Figure 5: Architecture Diagram**
- **Components:** Virchow2 encoder → Hist2ST decoder (CNN+Transformer+GNN) → Poisson output
- **Highlight:** Bias initialization (-3.0) and exp() activation
- **File:** CREATE NEW (TikZ or Inkscape)
- **LaTeX generation:** TikZ directly in manuscript or external SVG

### Supplementary Figures

**Figure S1: Fold Consistency**
- **File:** `figures/manuscript/figure_s1_fold_consistency.png` (ALREADY EXISTS)

**Figure S2-S4: Additional WSI Comparisons**
- **S2:** Secretory markers (MUC12, FCGBP)
- **S3:** Immune markers (JCHAIN, IGHA1)
- **S4:** Stromal/Other (VIM, LGALS3)
- **Files:** `figures/wsi_improved/*.png`

**Figure S5: Tile-Level Analysis**
- **Files:** `figures/tiles/*.png` (ALREADY EXISTS)

**Figure S6: Category Analysis**
- **File:** `figures/manuscript/figure_2_category_analysis.png` (ALREADY EXISTS)

### Tables

**Table 1 (Main Text): Model Performance Summary**
| Model | Decoder | Loss | SSIM 2μm | PCC 2μm | PCC 8μm |
|-------|---------|------|----------|---------|---------|
| E' | Hist2ST | Poisson | 0.542 ± 0.019 | 0.151 | 0.323 |
| F | Img2ST | Poisson | 0.268 ± 0.013 | ~0.00 | -0.004 |
| D' | Hist2ST | MSE | 0.200 ± 0.012 | 0.111 | 0.213 |
| G | Img2ST | MSE | 0.142 ± 0.007 | -0.006 | -0.039 |

**Table S1: Per-Gene Metrics**
- **File:** `tables/table_s1_pergene_metrics.csv` (ALREADY EXISTS)
- **Columns:** Gene, Category, Sparsity, MSE SSIM, Poisson SSIM, Δ-SSIM

**Table S2: Category Summary**
- **File:** `tables/table_s2_category_summary.csv` (ALREADY EXISTS)

**Table S3: Sparsity Quartiles**
- **File:** `tables/table_s3_sparsity_quartiles.csv` (ALREADY EXISTS)

---

## 3. Repository Structure (Target State)

```
mse-vs-poisson-2um-benchmark/
├── paper/                          # NEW: LaTeX manuscript
│   ├── main.tex                    # Main manuscript (IEEEtran or arXiv style)
│   ├── main.bib                    # References (BibTeX)
│   ├── supplementary.tex           # Supplementary materials
│   ├── figs/                       # Symlink to ../figures/manuscript/
│   └── build/                      # PDF outputs
│
├── src/                            # NEW: Clean source code
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoders.py             # Virchow2, future: Prov-GigaPath
│   │   ├── decoders.py             # Hist2ST, Img2ST
│   │   └── losses.py               # MSE, Poisson, future: ZINB
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py              # VisiumHDDataset
│   │   └── preprocessing.py        # Raw count extraction
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py              # Generic trainer
│   │   └── callbacks.py            # Early stopping, checkpointing
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py              # SSIM, PCC
│   │   └── visualization.py        # WSI, tile plots
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── config.py               # Hydra/OmegaConf configs
│
├── scripts/                        # REFACTORED: Entry points
│   ├── train_factorial.py          # Run full 2×2 factorial CV
│   ├── evaluate_model.py           # Evaluate single checkpoint
│   ├── generate_figures.py         # Regenerate all figures
│   └── reproduce_paper.sh          # One-click reproduction
│
├── tests/                          # NEW: Unit tests
│   ├── test_losses.py              # Test Poisson/MSE gradients
│   ├── test_models.py              # Test architectures
│   └── test_metrics.py             # Test SSIM/PCC calculation
│
├── configs/                        # NEW: Hydra configs
│   ├── model/
│   │   ├── hist2st_poisson.yaml
│   │   ├── hist2st_mse.yaml
│   │   ├── img2st_poisson.yaml
│   │   └── img2st_mse.yaml
│   ├── data/
│   │   └── visium_hd_crc.yaml
│   └── experiment/
│       └── factorial_cv.yaml
│
├── notebooks/                      # NEW: Analysis notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_sparsity_analysis.ipynb
│   └── 03_per_gene_analysis.ipynb
│
├── figures/                        # EXISTING: Keep as is
│   ├── manuscript/                 # Publication figures
│   ├── wsi_improved/               # WSI comparisons
│   └── tiles/                      # Tile comparisons
│
├── tables/                         # EXISTING: Keep as is
│
├── results/                        # EXISTING: Keep CV checkpoints
│
├── docs/                           # EXPANDED
│   ├── plans/                      # This file
│   ├── REPRODUCTION.md             # Step-by-step reproduction
│   └── ARCHITECTURE.md             # Model architecture details
│
├── README.md                       # UPDATED: Publication-oriented
├── LICENSE                         # NEW: MIT License
├── CITATION.cff                    # NEW: Citation metadata
├── requirements.txt                # NEW: Python dependencies
├── environment.yml                 # NEW: Conda environment
├── setup.py                        # NEW: Package installation
└── .zenodo.json                    # NEW: Zenodo metadata (DOI)
```

---

## 4. Implementation Phases

### Phase 1: Code Refactoring (Priority: High)
**Goal:** Clean, reusable, testable codebase

**Tasks:**
1. Extract model definitions from scripts into `src/models/`
2. Extract loss functions into `src/models/losses.py`
3. Create `src/data/dataset.py` for Visium HD loading
4. Build generic `src/training/trainer.py`
5. Write unit tests for critical components (losses, metrics)

**Success Criteria:**
- All scripts use imports from `src/`
- Tests pass with `pytest`
- Code coverage >70% for `src/models/losses.py`

---

### Phase 2: LaTeX Manuscript (Priority: High)
**Goal:** Complete, submission-ready manuscript

**Tasks:**
1. Create `paper/main.tex` with IEEEtran or arXiv template
2. Write all sections (Intro, Methods, Results, Discussion)
3. Generate all figure blocks using `mcp__latex-architect__generate_figure_block`
4. Create `paper/main.bib` with citations from `mcp__vanderbilt-professors__`
5. Create `paper/supplementary.tex`
6. Validate figure placement with `mcp__latex-architect__check_spatial_inclusion`

**Figure Generation (LaTeX Architect Standards):**
- All figures use `[t!]` placement (top-aligned, professional)
- Single-column: `0.95\columnwidth`
- Double-column: `0.95\textwidth` with `figure*`
- Non-breaking references: `Figure~\ref{fig:label}`

**Citations to Include:**
- Huo et al. (2025): Img2ST-Net (PMID:41210922) - baseline to improve upon
- Lau et al. (2022): CRC spatial atlas (PMID:35794563) - dataset context
- Sarkar et al. (2023/2025): Spatial methods - related work
- Foundation models: Virchow2, Prov-GigaPath, H-optimus-0 (future work)

**Success Criteria:**
- Compiles with `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
- No `[h]` or `[h!]` placements (verified by LaTeX Architect)
- All figures render correctly
- Word count ~6000-8000 (ArXiv has no limit, but aim for journal-length)

---

### Phase 3: Documentation (Priority: Medium)
**Goal:** Complete reproduction instructions

**Tasks:**
1. Write `docs/REPRODUCTION.md` with step-by-step commands
2. Write `docs/ARCHITECTURE.md` explaining model design
3. Update `README.md` for publication audience
4. Create `LICENSE` (MIT)
5. Create `CITATION.cff` with proper metadata
6. Create `requirements.txt` and `environment.yml`

**Success Criteria:**
- Fresh user can reproduce results with `reproduce_paper.sh`
- All dependencies documented
- Citation metadata parseable by GitHub/Zenodo

---

### Phase 4: Supplementary Materials (Priority: Medium)
**Goal:** Comprehensive supplementary document

**Tasks:**
1. Write `paper/supplementary.tex`
2. Include all supplementary figures (S1-S6)
3. Include all supplementary tables (S1-S3)
4. Add architectural details, hyperparameters, training curves
5. Add per-fold detailed results

**Success Criteria:**
- Supplementary compiles independently
- All figures/tables referenced
- Sufficient detail for full reproduction

---

### Phase 5: Testing & Validation (Priority: High)
**Goal:** Ensure reproducibility

**Tasks:**
1. Write unit tests for losses (Poisson gradient, MSE gradient)
2. Write unit tests for metrics (SSIM, PCC)
3. Write integration test: Train 1 epoch, verify loss decreases
4. Run `reproduce_paper.sh` on clean environment
5. Verify figure regeneration with `scripts/generate_figures.py`

**Success Criteria:**
- All tests pass: `pytest tests/ -v`
- Reproduction script completes without errors
- Generated figures match committed versions (visual inspection)

---

### Phase 6: ArXiv Submission Prep (Priority: High)
**Goal:** Ready for upload

**Tasks:**
1. Compile final PDF: `main.pdf` and `supplementary.pdf`
2. Create submission package (LaTeX source + figures)
3. Write ArXiv abstract (250 words)
4. Choose ArXiv categories: `q-bio.QM` (primary), `cs.CV`, `cs.LG`
5. Upload to ArXiv
6. Get ArXiv ID, update `CITATION.cff`

**Success Criteria:**
- ArXiv submission accepted
- PDF renders correctly on ArXiv viewer
- All figures display properly

---

### Phase 7: GitHub Release (Priority: Medium)
**Goal:** Citable, versioned release

**Tasks:**
1. Create GitHub release (v1.0.0)
2. Tag with ArXiv ID in release notes
3. Upload to Zenodo for DOI
4. Update `README.md` with ArXiv link and DOI
5. Update `CITATION.cff` with DOI

**Success Criteria:**
- GitHub release created with all assets
- Zenodo DOI obtained
- Repository citable

---

## 5. Key Citations (To Be Included)

### Primary Comparison
- **Huo et al. (2025)**: Img2ST-Net (PMID:41210922)
  - Uses MSE loss at 8-16μm
  - Achieves SSIM ~0.09 on CRC dataset
  - Our work shows Poisson is critical at 2μm (5.4× better SSIM)

### Spatial Transcriptomics Methods
- **Sarkar et al. (2023)**: Mapping spatial gene expression (PMID:37873258)
- **Sarkar et al. (2025)**: Joint imputation/deconvolution (PMID:41249066)

### CRC Spatial Biology
- **Lau et al. (2022)**: CRC spatial atlas (PMID:35794563)
  - Provides biological context for glandular structure
  - Validates importance of crypt architecture preservation

### Foundation Models
- **Virchow2**: Current encoder (cite preprint/repository)
- **Prov-GigaPath**: Future work (Nature 2024)
- **H-optimus-0**: Future work (Bioptimus 2024)

### Statistical Methods
- Factorial design references
- SSIM metric (Wang et al., IEEE TIP 2004)
- Poisson regression (McCullagh & Nelder, Generalized Linear Models)

---

## 6. Success Criteria

### Minimum Viable Publication
- ✅ LaTeX manuscript compiles
- ✅ 5 main figures + supplementary
- ✅ All tables included
- ✅ Code refactored into `src/`
- ✅ Basic tests pass
- ✅ README with reproduction instructions
- ✅ LICENSE and CITATION.cff

### Complete Publication Package
- ✅ All of above +
- ✅ Full test coverage (>70%)
- ✅ Comprehensive supplementary materials
- ✅ One-click reproduction script
- ✅ ArXiv uploaded
- ✅ GitHub release with DOI
- ✅ All figures use LaTeX Architect standards ([t!], 0.95 width)

### Stretch Goals
- ✅ Interactive notebooks for exploration
- ✅ Docker container for reproduction
- ✅ Pre-commit hooks for code quality
- ✅ Automated figure regeneration CI

---

## 7. Timeline Estimate

| Phase | Estimated Time | Priority |
|-------|----------------|----------|
| Code Refactoring | 6-8 hours | High |
| LaTeX Manuscript | 8-10 hours | High |
| Documentation | 3-4 hours | Medium |
| Supplementary | 4-5 hours | Medium |
| Testing | 4-6 hours | High |
| ArXiv Prep | 2-3 hours | High |
| GitHub Release | 1-2 hours | Medium |
| **Total** | **28-38 hours** | **1.5-2 days focused work** |

---

## 8. Risk Mitigation

### Risk 1: Figure Quality Issues
- **Mitigation:** Use LaTeX Architect MCP for all figure blocks
- **Validation:** Check spatial inclusion before finalizing

### Risk 2: Reproducibility Failures
- **Mitigation:** Write tests early, run reproduction script frequently
- **Validation:** Clean environment test before submission

### Risk 3: Citation Errors
- **Mitigation:** Use Vanderbilt Professors MCP for accurate PMIDs
- **Validation:** CrossRef/PubMed verification of all citations

### Risk 4: LaTeX Compilation Issues
- **Mitigation:** Use standard templates (IEEEtran or ArXiv base)
- **Validation:** Test compilation after each section

---

## 9. Post-Publication Plans

### Immediate Next Steps
1. **Venue Submission**: Adapt for MICCAI 2026 or Bioinformatics
2. **ZINB Implementation**: Test H_011 and H_013 hypotheses from Polymax
3. **Multi-tissue Validation**: Test on breast, lung, other cancers

### Long-term Research Trajectory
1. **Better Encoders**: Prov-GigaPath, H-optimus-0 (cancer-optimized)
2. **Better Losses**: ZINB for zero-inflation + overdispersion
3. **Better Architectures**: TICON tile contextualizer (if weights released)
4. **Scaling**: Whole-slide inference, multi-patient generalization

---

## 10. Design Decisions Summary

✅ **Narrative:** Hybrid (Sparsity Trap + Factorial Validation)
✅ **Title:** "The Sparsity Trap: Why MSE Fails and Poisson Succeeds for 2μm Spatial Transcriptomics Prediction"
✅ **Target Venue:** ArXiv (adaptable to MICCAI/Bioinformatics)
✅ **Main Contribution:** Loss function is critical for sparse ST
✅ **Secondary Contribution:** Factorial validation of decoder + loss
✅ **Figure Strategy:** 5 main + 6 supplementary + 3 tables
✅ **Code Structure:** Modular `src/` with tests
✅ **LaTeX Standards:** IEEE/MICCAI formatting via LaTeX Architect MCP

---

**Status:** Design validated and ready for implementation planning.

**Next Steps:**
1. Create git worktree for isolated workspace
2. Write detailed implementation plan with `/write-plan` skill
3. Execute plan in batches with `/execute-plan` skill

---

**End of Design Document**
