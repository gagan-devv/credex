# Credex: Cost-Sensitive & Explainable Machine Learning Platform for Credit Risk Assessment

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **An Asymmetric AI Extension, Empirical Benchmark Enhancement, and Research Paper Generation Platform Based on Li & Wu (2024).**

---

## 1. Executive Summary & Research Vision

**Credex** is an end-to-end, production-grade Machine Learning and Financial Economics platform for consumer credit default risk assessment. It is engineered to replicate, empirically challenge, and significantly enhance the findings of the benchmark research paper:

> **Li, H., & Wu, W. (2024). "Loan default predictability with explainable machine learning." *Finance Research Letters*, 60, Article 104867.**

While Li & Wu (2024) established strong baseline predictability across nine machine learning algorithms on LendingClub data (reporting champion ROC-AUC of 0.835 with Random Forest and 0.831 with LightGBM), their implementation embodies critical theoretical and operational shortcomings:

1. **Symmetric Loss Formulation:** Treating False Positives (denying a creditworthy borrower) and False Negatives (granting credit to a borrower who defaults) as having equal cost ($1:1$), ignoring that credit default write-offs typically cost $5\times$ to $10\times$ more than foregone net interest margin.
2. **Arbitrary Decision Thresholds:** Defaulting to an uncalibrated 0.50 probability cutoff rather than calculating the cost-optimal cutoff ($\tau^*$) that minimizes expected portfolio credit loss.
3. **Endogeneity & Pricing Leakage:** Using downstream lender-assigned variables (`interest_rate`, `sub_grade`) to predict default, introducing circularity into credit risk estimation.
4. **Passive Explainability vs. Actionable Recourse:** Confining Explainable AI (XAI) to post-hoc global feature importance charts without generating legally compliant, actionable adverse action disclosures under the Equal Credit Opportunity Act (ECOA, Regulation B) and Fair Credit Reporting Act (FCRA).

Credex addresses each of these gaps, providing a complete empirical ablation engine that automatically compiles publication-ready LaTeX tables, 300-DPI vector figures, and statistical digests for direct inclusion in a new academic research paper.

---

## 2. Core Methodological Enhancements

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   CORE METHODOLOGY                                      │
├───────────────────────────────┬────────────────────────────────────────────────────────┤
│ Dimension                     │ Li & Wu (2024) Baseline  │ Credex Enhanced Platform     │
├───────────────────────────────┼──────────────────────────┼─────────────────────────────┤
│ Objective Function            │ Symmetric Cross-Entropy  │ Custom Asymmetric Gradients │
│ Decision Cutoff (τ)           │ Fixed at 0.50            │ Cost-Optimal Sweep (τ*)     │
│ Loss Evaluation               │ Statistical (AUC/Acc)    │ Unit Matrix + Dollar Loss   │
│ Feature Space                 │ 15 Raw Features          │ Baseline + Solvency Ratios  │
│ Endogeneity Handling          │ Pricing Features Kept    │ Explicit Pricing Ablation   │
│ Imbalance Treatment           │ Standard Imbalance       │ SMOTE-Tomek Hybrid Resample │
│ Explainability (XAI)          │ Global Summary Plots     │ TreeSHAP + Actionable Recourse│
│ Evaluation Protocol           │ 5-Fold Stratified CV     │ 5-Fold CV + Out-of-Time OOT │
│ Automated Output              │ Manual Inspection        │ Direct LaTeX & Vector PDFs  │
└───────────────────────────────┴──────────────────────────┴─────────────────────────────┘
```

---

## 3. Repository Architecture

```
credex/
├── README.md                           # Quickstart guide, benchmark summary, CLI usage
├── pyproject.toml                      # Modern packaging and tool configuration
├── requirements.txt                    # Pinned production dependencies
├── setup.py                            # Package installer
├── Makefile                            # One-touch targets (install, lint, test, run)
├── configs/
│   ├── base_config.yaml                # Random seed, sample sizes, paths, OOT dates
│   ├── features_config.yaml            # Definitions for baseline vs enhanced feature sets
│   ├── models_config.yaml              # Hyperparameters for all 9 models
│   └── cost_matrix.yaml                # Scalar penalties (5:1, 10:1), LGD, NIM parameters
├── src/
│   ├── __init__.py
│   ├── cli.py                          # Centralized CLI orchestrator & subcommands
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                   # Ingestion for LendingClub & South German Credit
│   │   ├── preprocessor.py             # Out-of-fold imputation, scaling, winsorization
│   │   ├── feature_engineering.py      # Solvency ratios & pricing feature toggle
│   │   └── resampler.py                # SMOTE-Tomek & balanced sampling pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baselines.py                # Replication of 9 Li & Wu algorithms
│   │   ├── cost_sensitive.py           # Custom asymmetric gradient LightGBM/CatBoost/XGB
│   │   ├── objective_functions.py      # 1st & 2nd order loss gradients (g_i, h_i)
│   │   ├── calibration.py              # Isotonic regression & Platt scaling calibration
│   │   └── threshold_optimizer.py      # Optimal tau* sweep across expected loss curves
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── shap_explainer.py           # Fast TreeSHAP computation engine
│   │   ├── visualizer.py               # Beeswarm, interaction, and waterfall plots
│   │   └── adverse_action.py           # Constrained counterfactual recourse generator
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py                  # Statistical & financial portfolio loss metrics
│       ├── ablation_runner.py          # 2x3x2 factorial ablation execution engine
│       └── paper_artifacts.py          # Automated LaTeX table & vector figure compiler
├── experiments/
│   └── runs/                           # Timestamped run outputs, metrics JSONs, CSVs
├── reports/
│   └── paper/
│       ├── PAPER_RESULTS_DIGEST.md     # Executive summary with paper numbers & p-values
│       ├── tables/                     # Ready-to-include LaTeX tables (.tex)
│       └── figures/                    # 300-DPI vector publication figures (.pdf, .png)
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb # Interactive distribution inspection
│   ├── 02_baseline_replication.ipynb      # Reproduction verification of Li & Wu
│   └── 03_explainability_and_recourse.ipynb # Interactive SHAP and recourse demonstrations
└── tests/
    ├── test_data_leakage.py            # Automated tests ensuring no future repayment data
    ├── test_gradients.py               # Numerical finite-difference vs. analytical gradient test
    ├── test_threshold_optimizer.py     # Expected loss minimization correctness test
    └── test_counterfactual_recourse.py # Verifies immutable features remain unaltered
```

---

## 4. Technology Stack & Core Dependencies

The platform leverages modern, industry-standard data science and financial machine learning libraries:

| Layer | Component | Version | Role in Platform |
| :--- | :--- | :--- | :--- |
| **Language** | Python | `3.10+` | Core environment for typed, reproducible execution. |
| **Data Processing** | `pandas`, `numpy` | `2.1+`, `1.24+` | Out-of-fold preprocessing, winsorization, vector math. |
| **Scientific Computing** | `scipy` | `1.11+` | Statistical significance testing ($t$-test, Mann-Whitney). |
| **Machine Learning** | `scikit-learn` | `1.3+` | 9 baseline algorithms, cross-validation, calibration. |
| **Gradient Boosting** | `lightgbm`, `xgboost`, `catboost` | `4.1+`, `2.0+`, `1.2+` | Asymmetric custom loss gradients ($g_i, h_i$) & fast training. |
| **Imbalance Remediation** | `imbalanced-learn` | `0.11+` | Synthetic minority oversampling and Tomek link cleaning. |
| **Explainable AI (XAI)** | `shap` | `0.44+` | TreeSHAP exact attributions & interaction values. |
| **Hyperparameter Tuning** | `optuna` | `3.4+` | Bayesian TPE optimization across validation folds. |
| **Vector Graphics** | `matplotlib`, `seaborn` | `3.8+`, `0.13+` | Publication-ready vector figures (PDF/PNG at 300+ DPI). |
| **Configuration** | `pyyaml`, `pydantic` | `6.0+`, `2.5+` | Strict type validation for financial parameters and cost matrices. |
| **Testing** | `pytest` | `7.4+` | Leakage prevention, gradient validation, recourse assertions. |

---

## 5. Unified Command-Line Interface (CLI)

The platform is driven through a single modular CLI entry point:

```bash
# 1. Ingest, preprocess, and partition datasets
python -m src.cli ingest --dataset lending_club --sample-size 200000 --oot-split

# 2. Replicate Li & Wu (2024) 9 baseline algorithms and verify Table 2
python -m src.cli replicate-paper

# 3. Execute 2x3x2 Factorial Ablation Matrix across cost ratios (5:1, 10:1)
python -m src.cli run-ablation --models lightgbm,catboost,xgboost

# 4. Generate TreeSHAP attributions & actionable counterfactual recourses
python -m src.cli explain --sample-n 5000 --recourse-applicants 10

# 5. Compile LaTeX tables, 300-DPI vector figures, and paper results digest
python -m src.cli export-paper --output-dir reports/paper

# 6. One-Shot Execution: Runs end-to-end pipeline from data to paper outputs
python -m src.cli run-all --sample-size 200000
```

---

## 6. Implementation & Research Roadmap

- [ ] **Day 1: Ingestion & Validation Pipelines:** Load LendingClub / German Credit, out-of-fold preprocessing, solvency ratio engineering, target leakage testing.
- [ ] **Day 2: Baseline Replication Suite:** Train all 9 algorithms under standard cross-entropy; reproduce Table 2 of Li & Wu (2024).
- [ ] **Day 3: Asymmetric Loss & Threshold Tuning:** Implement custom gradients ($g_i, h_i$), numerical gradient verification, probability calibration, expected loss optimizer ($\tau^*$).
- [ ] **Day 4: Factorial Ablation Matrix:** SMOTE-Tomek integration, execute $2 \times 3 \times 2$ grid across 5-fold CV and Out-of-Time holdouts.
- [ ] **Day 5: Explainability & Actionable Recourse:** TreeSHAP beeswarm/interaction plots, constrained counterfactual recourse solver with immutable feature freezing.
- [ ] **Day 6: Paper Artifact Compilation:** Automated compilation of LaTeX tables (`table1`–`table4`), vector figures (`fig1`–`fig4`), and `PAPER_RESULTS_DIGEST.md`.
- [ ] **Day 7: Paper Drafting & Final Freeze:** Draft academic paper sections from generated artifacts, run end-to-end unit tests, final code package freeze.

---

## 7. Team & Collaboration

This repository serves as the core computational and empirical foundation for a 4-person research team. Detailed technical specifications, mathematical derivations, and acceptance criteria are documented in the project's PRD for subsequent work breakdown and sprint ticket assignment.

---

## 8. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
