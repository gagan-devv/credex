# Product Requirements Document (PRD)

## Project Title

**Cost-Sensitive & Explainable Machine Learning Platform for Credit Risk Assessment**  
*An Asymmetric AI Extension, Empirical Benchmark Enhancement, and Research Paper Generation Engine Based on Li & Wu (2024)*

---

## 1. Executive Summary & Research Vision

### 1.1 Project Overview
This project delivers a dual-purpose, production-grade financial machine learning platform and empirical research engine for retail loan default prediction. It directly addresses the theoretical, econometric, and regulatory deficiencies of the published benchmark paper:

> **Li, H., & Wu, W. (2024). "Loan default predictability with explainable machine learning." *Finance Research Letters*, 60, Article 104867.**

While Li & Wu (2024) demonstrated strong default predictability using tree-based ensemble models (reporting a top ROC-AUC of 0.835 with Random Forest and 0.831 with LightGBM on LendingClub data), their framework embodies four critical limitations that undermine real-world banking deployment and econometric validity:

1. **Symmetric Loss Formulation:** Treats False Positives (denying a creditworthy borrower) and False Negatives (granting credit to a defaulting borrower) with equal penalty, ignoring that retail banking write-offs are typically $5\times$ to $10\times$ more costly than foregone net interest margin.
2. **Arbitrary Classification Cutoff:** Defaults to an uncalibrated $\tau = 0.5$ probability threshold rather than determining the cost-optimal decision threshold ($\tau^*$) that minimizes expected portfolio credit losses.
3. **Endogeneity & Circular Pricing Leakage:** Includes lender-assigned pricing features (`interest_rate`, `sub_grade`) that are determined downstream by the platform's proprietary risk committee, introducing econometric endogeneity into default prediction.
4. **Passive Feature Attribution vs. Actionable Recourse:** Confines Explainable AI (XAI) to post-hoc global feature importance, failing to deliver legally mandated, actionable adverse action notices under the Equal Credit Opportunity Act (ECOA, Regulation B) and Fair Credit Reporting Act (FCRA).

### 1.2 Dual Project Deliverables
To fulfill both engineering excellence and academic research publication within a **strict 7-day delivery horizon**, the project is architected around two primary deliverables:
1. **The Software Platform (`credex`):** A modular, production-grade Python package with an automated Command Line Interface (CLI) executing data ingestion, feature engineering, baseline replication, asymmetric gradient training, factorial ablation, and TreeSHAP counterfactual recourse.
2. **The Research Paper Artifacts:** An automated reporting pipeline that compiles empirical benchmark outputs directly into publication-ready LaTeX tables, 300-DPI vector figures, and an executive results digest formatted for submission to a peer-reviewed finance/AI journal (e.g., *Finance Research Letters*, *Expert Systems with Applications*).

---

## 2. Core Methodological Enhancements over Li & Wu (2024)

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

## 3. Data Architecture & Preprocessing Engine

### 3.1 Supported Datasets
The platform supports two complementary datasets ensuring empirical robustness across geographies and loan types:

1. **LendingClub Consumer Loan Dataset (Primary Academic Benchmark):**
   - **Population Scope:** Completed loan contracts with definitive terminal performance (`loan_status` $\in$ {`Fully Paid` $\rightarrow 0$, `Charged Off` / `Default` $\rightarrow 1$}). Active, late, or in-grace-period loans are strictly excluded.
   - **Scale Modes:**
     - *Development / Rapid Tuning Mode:* Deterministic stratified cohort of 200,000 completed loans (fixed random seed).
     - *Full Benchmark Mode:* Entire historical cohort of ~1.3 million completed loans spanning 2007–2018Q4 (`--full-dataset` flag).
2. **UCI South German Credit Dataset (Secondary Standardization Benchmark):**
   - Standardized 1,000-sample commercial credit dataset featuring a peer-reviewed $5:1$ misclassification cost matrix ($C_{FN}:C_{FP} = 5:1$).

### 3.2 Dual Validation Partitioning Protocol
To satisfy academic benchmark replication while testing macroeconomic resilience:
- **Partition 1: Stratified 5-Fold Cross-Validation:** Replicates the in-sample cross-validation methodology of Li & Wu (2024), preserving the empirical ~20% default rate across each partition.
- **Partition 2: Out-of-Time (OOT) Holdout Partition:** Vintages 2012–2016 serve as the historical development window; vintages 2017–2018 serve as the strictly out-of-time evaluation window to evaluate model degradation under temporal and macroeconomic regime shifts.

### 3.3 Dual Feature Regimes & Endogeneity Isolation
The platform establishes two explicit feature configurations managed via YAML configs:

1. **`feature_set_baseline` (Exact Replication of Li & Wu 2024):**
   - Solvency / Leverage: `debt_to_income` (`dti`), `annual_income`
   - Contract Terms: `loan_amount`, `installment`, `interest_rate`, `term`
   - Credit Profile: `fico_score` / `grade` / `sub_grade`, `delinq_2yrs`, `revol_utilization`, `inq_last_6mths`, `open_acc`, `pub_rec`, `revol_bal`, `total_acc`
   - Demographics: `employment_length` (`emp_length`), `home_ownership`
2. **`feature_set_enhanced` (Domain Solvency Ratios & Endogeneity Ablation):**
   - Includes all baseline variables augmented with engineered economic indicators:
     - **Payment-to-Income (PTI):** $\text{installment} / (\text{annual\_income} / 12)$
     - **Loan-to-Income (LTI):** $\text{loan\_amount} / \text{annual\_income}$
     - **Available Liquidity Buffer:** $\text{total\_rev\_hi\_lim} - \text{revol\_bal}$
     - **Delinquency Velocity:** $\text{delinq\_2yrs} / \text{total\_acc}$
   - **Econometric Pricing Ablation:** Controlled toggle (`include_pricing_features: false`) that removes lender-assigned `interest_rate`, `grade`, and `sub_grade` to evaluate intrinsic borrower default risk free from downstream platform circularity.

### 3.4 Leakage Prevention & Preprocessing Standards
- **Strict Out-of-Fold Preprocessing:** Imputation statistics (median for numericals, mode for categoricals), standard scalers, and target encodings are fitted strictly on training folds and transformed onto test folds.
- **Financial Winsorization:** Continuous financial variables (`annual_income`, `revol_bal`, `dti`) are winsorized at the 99.5th percentile rather than truncated, preserving high-value loan profiles while mitigating gradient explosion in tree models.
- **Target Leakage Auditing:** Unit tests (`tests/test_data_leakage.py`) programmatically verify that post-origination repayment fields (`recoveries`, `collection_recovery_fee`, `last_pymnt_amnt`, `total_rec_prncp`) are purged prior to training.

---

## 4. Modeling, Optimization & Factorial Ablation

### 4.1 Replication Suite (9 Baseline Algorithms)
Replicates Table 2 of Li & Wu (2024) under standard binary cross-entropy loss with default classification threshold $\tau = 0.5$:
1. Logistic Regression ($L_2$ penalty)
2. Decision Tree Classifier (CART, tuned depth)
3. Random Forest Classifier (Paper benchmark champion: AUC 0.835)
4. AdaBoost Classifier (SAMME.R)
5. Gradient Boosting Decision Tree (GBDT / Scikit-Learn)
6. XGBoost (Extreme Gradient Boosting)
7. LightGBM (Light Gradient Boosting Machine, paper runner-up: AUC 0.831)
8. CatBoost (Categorical Boosting)
9. Multilayer Perceptron (MLP Neural Network, 2 hidden layers, ReLU, Adam)

### 4.2 Asymmetric Cost-Sensitive Objective Formulation
Standard binary cross-entropy treats errors symmetrically:

$$\mathcal{L}_{\text{sym}}(y, p) = - [y \log(p) + (1-y) \log(1-p)]$$

The platform implements an asymmetric cost-sensitive loss function where false negatives carry a penalty multiplier $w = C_{FN} / C_{FP} \ge 5$:

$$\mathcal{L}_{\text{cost}}(y, p) = - [w \cdot y \log(p) + (1-y) \log(1-p)]$$

For gradient boosted trees (LightGBM, XGBoost, CatBoost), custom first- and second-order loss derivatives with respect to raw tree margin $\hat{y}_i$ (where $p_i = \sigma(\hat{y}_i) = \frac{1}{1 + e^{-\hat{y}_i}}$) are derived and supplied directly:

$$g_i = \frac{\partial \mathcal{L}_{\text{cost}}}{\partial \hat{y}_i} = p_i (1 + (w - 1) y_i) - w y_i$$

$$h_i = \frac{\partial^2 \mathcal{L}_{\text{cost}}}{\partial \hat{y}_i^2} = p_i (1 - p_i) (1 + (w - 1) y_i)$$

### 4.3 Probability Calibration
Because custom loss gradients shift raw model outputs into cost-skewed margins rather than true posterior probabilities $P(Y=1|X)$, the platform applies **Isotonic Regression** and **Platt Scaling** on cross-validation holdouts before thresholding, ensuring probability calibration for risk scoring and Brier score evaluation.

### 4.4 Economic Threshold Optimizer
Rather than imposing an arbitrary $\tau = 0.5$ cutoff, the platform executes a fine-grained threshold sweep $\tau \in [0.01, 0.99]$ (resolution $\Delta \tau = 0.005$) to find the empirical minimum expected loss:

$$\tau^* = \arg\min_{\tau \in (0, 1)} \mathbb{E}[\text{Loss}(\tau)] = \arg\min_{\tau} \left[ C_{FN} \cdot FN(\tau) + C_{FP} \cdot FP(\tau) \right]$$

### 4.5 Dual Economic Loss Framework
The platform quantifies financial performance under two distinct criteria:

1. **Standardized Unit Cost Matrix ($C_{FN}:C_{FP}$):**
   - Evaluated across canonical ratios: $5:1$ (South German benchmark) and $10:1$ (Retail banking stress scenario).
   - Metric: Normalized Expected Unit Loss per applicant.
2. **Instance-Dependent Portfolio Dollar Loss ($\Delta \$ \text{Loss}$):**
   - Incorporates loan-level contract exposure:
     - Default Write-Off Loss: $C_{FN, i} = \text{loan\_amount}_i \times \text{LGD}$ (where $\text{LGD} \in [0.60, 0.75]$, calibrated to LendingClub historical charge-off recoveries).
     - Rejection Opportunity Cost: $C_{FP, i} = \text{loan\_amount}_i \times \text{NIM} \times \frac{\text{term}}{12}$ (where $\text{NIM} = 0.035$, representing net interest margin).
   - Total Portfolio Dollar Loss:

$$\text{Loss}_{\text{portfolio}}(\tau) = \sum_{i \in FN(\tau)} (\text{loan\_amount}_i \times \text{LGD}) + \sum_{j \in FP(\tau)} \left( \text{loan\_amount}_j \times \text{NIM} \times \frac{\text{term}_j}{12} \right)$$

### 4.6 Structured Factorial Ablation Matrix ($2 \times 3 \times 2$)
To isolate which algorithmic intervention drives performance in the research paper, the top tree algorithms (LightGBM, CatBoost, XGBoost, Random Forest) are evaluated across all 12 combinations of:
- **Sampling Dimension (2):** Natural Imbalance vs. SMOTE-Tomek Hybrid Resampling
- **Objective Dimension (3):** Standard Cross-Entropy vs. Class-Weighted Loss (`scale_pos_weight`) vs. Custom Asymmetric Gradients ($g_i, h_i$)
- **Decision Cutoff Dimension (2):** Default $\tau = 0.5$ vs. Cost-Optimal $\tau^*$

---

## 5. Explainability, Governance & Actionable Recourse

### 5.1 TreeSHAP Global & Interaction Attributions
- Computes exact local and global Shapley additive attributions using C++ optimized `TreeExplainer` on test partitions ($N \le 5,000$ to maintain sub-minute runtimes):

$$f(x) = \phi_0 + \sum_{j=1}^{M} \phi_j(x)$$

- Generates global beeswarm summary plots and pairwise dependence interaction plots (e.g., `dti` $\times$ `annual_income`, `revol_utilization` $\times$ `fico_score`).

### 5.2 Actionable Counterfactual Recourse Engine
To comply with ECOA Regulation B and CFPB adverse action mandates, the system moves beyond passive feature attributions by generating actionable, feasible recourses for rejected applicants ($p_i \ge \tau^*$):

1. **Feature Mutability Constraints:**
   - **Mutable Features:** Requested `loan_amount`, revolving balance / utilization (`revol_utilization`), scheduled installment (via term adjustment).
   - **Immutable Features:** Historical delinquencies (`delinq_2yrs`), credit history length, past inquiries, public records.
2. **Constrained Recourse Solver:**
   - Evaluates minimal $L_1$-norm perturbations $\delta$ on normalized mutable variables $X_{\text{mut}}$ such that the updated prediction satisfies:

$$f(X_{\text{mut}} + \delta, X_{\text{immut}}) < \tau^*$$

   - Implemented via a vectorized grid-projection solver ensuring execution times $< 50$ milliseconds per applicant.
3. **Automated Regulatory Adverse Action Notice:**
   - Formats the top 3 adverse TreeSHAP factors alongside the minimal counterfactual recourse into standardized plain-language compliance text:
   ```
   ================================================================================
   ADVERSE ACTION DISCLOSURE NOTICE (REGULATION B / ECOA COMPLIANT)
   ================================================================================
   Application ID: LC-9842104                    Decision: REJECTED (Risk: 34.2% >= Cutoff: 21.5%)
   
   PRINCIPAL ADVERSE FACTORS CONTRIBUTING TO RISK SCORE:
   1. Debt-to-Income Ratio (38.4%) contributed +1.42 to default log-odds.
   2. Revolving Credit Line Utilization (82.1%) contributed +0.94 to default log-odds.
   3. Recent 30+ Day Delinquencies (2 occurrences) contributed +0.65 to default log-odds.
   
   ACTIONABLE COUNTERFACTUAL RECOURSE (FEASIBLE APPROVAL PATH):
   To reduce default risk below the approval threshold (tau* = 0.215), the applicant may:
     [Option A] Reduce requested loan amount from $25,000 to $18,500 (-$6,500).
     [Option B] Pay down revolving credit card balances from $14,200 to $8,100 (-$6,100).
   ================================================================================
   ```

---

## 6. Automated Research Paper Artifact Generation

To eliminate manual transcription and format results directly for publication drafting, `src/evaluation/paper_artifacts.py` automatically compiles all benchmark runs into publication-ready files stored under `reports/paper/`:

### 6.1 Generated LaTeX Tables (`reports/paper/tables/`)
1. **`table1_descriptive_stats.tex`:** Summary statistics (Mean, SD, Median, Min, Max, Two-sample $t$-test / Mann-Whitney $U$ test $p$-values comparing Default vs. Non-Default cohorts).
2. **`table2_li_wu_replication.tex`:** Direct replication table comparing all 9 models against published Li & Wu (2024) numbers across ROC-AUC, Accuracy, Precision, Recall, and F1-Score.
3. **`table3_cost_sensitive_ablation.tex`:** Factorial ablation table ($2 \times 3 \times 2$) displaying ROC-AUC, PR-AUC, Default Recall, Expected Unit Loss ($5:1$ and $10:1$), and Brier Score, with best-in-column results automatically bolded.
4. **`table4_economic_dollar_savings.tex`:** Total portfolio charge-off losses, foregone interest costs, net economic loss, and dollar savings ($\Delta \$$ and $\%$) compared to the Li & Wu baseline across different LGD assumptions ($60\%, 70\%, 75\%$).

### 6.2 Publication-Grade Vector Figures (`reports/paper/figures/`)
All figures rendered at 300+ DPI in both vector PDF (for LaTeX `\includegraphics`) and PNG (for presentation):
1. **`fig1_roc_pr_comparison.pdf`:** Dual-panel ROC and Precision-Recall curves comparing Li & Wu baseline vs. Credex Cost-Sensitive Champion.
2. **`fig2_expected_loss_curves.pdf`:** Expected dollar loss curves across threshold $\tau \in [0.01, 0.99]$, contrasting the arbitrary $\tau = 0.5$ point against the cost-optimal cutoff $\tau^*$.
3. **`fig3_shap_global_beeswarm.pdf`:** Exact TreeSHAP global beeswarm plot of feature attributions across the top 15 risk drivers.
4. **`fig4_actionable_recourse_waterfall.pdf`:** Dual-panel local visualization: Left = TreeSHAP rejection waterfall; Right = Counterfactual recourse trajectory shifting probability below $\tau^*$.

### 6.3 Executive Results Digest (`reports/paper/PAPER_RESULTS_DIGEST.md`)
A markdown briefing containing exact metric deltas, percentage lifts, and statistical significance statements ($p < 0.001$), providing ready-to-paste text for the Abstract, Empirical Results, and Conclusion sections of the paper.

---

## 7. Repository Structure & Code Architecture

```
credex/
├── README.md                           # Quickstart guide, benchmark results, CLI usage
├── pyproject.toml                      # Modern packaging and tool configs
├── requirements.txt                    # Pinned production dependencies
├── setup.py                            # Editable package installer
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
├── tests/
│   ├── test_data_leakage.py            # Automated tests ensuring no future repayment data
│   ├── test_gradients.py               # Numerical finite-difference vs. analytical gradient test
│   ├── test_threshold_optimizer.py     # Expected loss minimization correctness test
│   └── test_counterfactual_recourse.py # Verifies immutable features remain unaltered
├── docs/
│   ├── adr/                            # Architectural Decision Records (0001–0006)
│   └── agents/                         # Agent skill configurations and triage specs
└── GLOSSARY.md                         # Ubiquitous domain language definitions
```

---

## 8. Technology Stack & Pinned Dependencies

| Layer | Library / Tool | Pinned Version | Economic & Engineering Rationale |
| :--- | :--- | :--- | :--- |
| **Language** | Python | `3.10+` | Core standard for financial ML, scientific libraries, and typed dataclasses. |
| **Data Processing** | `pandas` | `>=2.1.0,<2.3.0` | High-performance tabular manipulation, PyArrow backend support. |
| **Numerical Math** | `numpy`, `scipy` | `>=1.24.0`, `>=1.11.0` | Vectorized ratio calculations, numerical gradient checks, statistical tests. |
| **Standard ML** | `scikit-learn` | `>=1.3.0,<1.5.0` | 9 baseline algorithms, cross-validation splits, calibration, ROC/PR metrics. |
| **Gradient Boosting**| `lightgbm` | `>=4.1.0` | Native custom objective callbacks (`fobj`), blazing fast tree building. |
| **Gradient Boosting**| `xgboost` | `>=2.0.0` | Robust secondary tree ensemble with custom Hessian evaluation. |
| **Gradient Boosting**| `catboost` | `>=1.2.0` | Native categorical handling, asymmetric tree growth support. |
| **Imbalance Engine** | `imbalanced-learn`| `>=0.11.0` | Standardized implementation of SMOTE and Tomek link elimination. |
| **Explainability** | `shap` | `>=0.44.0` | High-speed C++ `TreeExplainer` for exact additive feature attributions. |
| **Hyperparameter** | `optuna` | `>=3.4.0` | Bayesian optimization (TPE) for tuning tree depth, regularization, learning rates. |
| **Visualization** | `matplotlib`, `seaborn` | `>=3.8.0`, `>=0.13.0` | Publication-grade vector graphics (PDF/PNG at 300+ DPI), LaTeX font rendering. |
| **Configuration** | `pyyaml`, `pydantic` | `>=6.0.1`, `>=2.5.0` | Type-safe configuration schemas for financial parameters and cost matrices. |
| **Artifact Tables** | `tabulate`, `jinja2` | `>=0.9.0`, `>=3.1.0` | Programmatic formatting of publication-ready LaTeX tables. |
| **Testing** | `pytest` | `>=7.4.0` | Unit testing framework for leakage prevention, gradients, and recourse logic. |

---

## 9. Unified CLI Specification & Execution Workflows

The platform is operated via a single entrypoint `src/cli.py` designed for modularity and headless execution:

### 9.1 CLI Subcommands
```bash
# 1. Ingest, clean, winsorize, and partition data (Dual-mode sample or full)
python -m src.cli ingest --dataset lending_club --sample-size 200000 --oot-split

# 2. Replicate Li & Wu (2024) 9 baseline models and verify Table 2 metrics
python -m src.cli replicate-paper --output-dir experiments/runs/baseline_replication

# 3. Execute 2x3x2 Factorial Ablation Matrix (Custom Gradients vs Thresholds vs SMOTE)
python -m src.cli run-ablation --models lightgbm,catboost,xgboost --cost-ratios 5,10

# 4. Generate TreeSHAP explanations and counterfactual recourses for test applicants
python -m src.cli explain --sample-n 5000 --recourse-applicants 10

# 5. Compile all empirical outputs into LaTeX tables and 300-DPI vector figures
python -m src.cli export-paper --output-dir reports/paper

# 6. MASTER ONE-SHOT EXECUTION: Executes full pipeline from raw data to paper artifacts
python -m src.cli run-all --sample-size 200000
```

---

## 10. Seven-Day Implementation & Paper Deliverable Roadmap

```
Day 1: Repo Setup, Data Ingestion & Dual-Feature Pipelines
├─ Initialize directory structure, pyproject.toml, Makefile, and pinned requirements.txt
├─ Implement src/data/loader.py (LendingClub & South German Credit ingestion)
├─ Implement src/data/preprocessor.py (Out-of-fold imputation, scaling, winsorization)
├─ Implement src/data/feature_engineering.py (Solvency ratios & pricing feature toggle)
└─ Implement tests/test_data_leakage.py and verify clean 5-fold CV & OOT partitions

Day 2: Baseline Replication Suite (Li & Wu 2024 Table 2)
├─ Implement src/models/baselines.py (9 algorithms under symmetric cross-entropy)
├─ Run baseline benchmark across Stratified 5-Fold Cross-Validation
├─ Validate that Random Forest (AUC ~0.835) and LightGBM (AUC ~0.831) match published values
└─ Log baseline run artifacts to experiments/runs/baseline_replication.json

Day 3: Asymmetric Custom Loss Gradients & Threshold Optimizer
├─ Implement src/models/objective_functions.py (Analytical 1st & 2nd order gradients g_i, h_i)
├─ Implement tests/test_gradients.py (Finite-difference numerical gradient validation)
├─ Implement src/models/cost_sensitive.py (LightGBM/CatBoost/XGBoost custom objective wrappers)
├─ Implement src/models/calibration.py (Isotonic regression & Platt scaling)
└─ Implement src/models/threshold_optimizer.py (Expected loss minimization sweep across tau)

Day 4: Imbalance Remediation & Factorial Ablation Matrix
├─ Implement src/data/resampler.py (SMOTE-Tomek balanced sampling pipeline)
├─ Implement src/evaluation/metrics.py (Statistical metrics + Dollar Portfolio Loss)
├─ Implement src/evaluation/ablation_runner.py (Executing 2x3x2 factorial grid)
├─ Run ablation suite across 5-Fold CV and OOT holdout partition
└─ Persist full ablation metrics to experiments/runs/ablation_benchmark.json

Day 5: Explainability Engine & Actionable Recourse Generator
├─ Implement src/explainability/shap_explainer.py (TreeSHAP attributions)
├─ Implement src/explainability/visualizer.py (Beeswarm, interaction, and waterfall plots)
├─ Implement src/explainability/adverse_action.py (Constrained counterfactual recourse solver)
├─ Implement tests/test_counterfactual_recourse.py (Verifying immutable feature preservation)
└─ Generate compliance-ready adverse action disclosure notices

Day 6: Automated Research Paper Artifact Compilation
├─ Implement src/evaluation/paper_artifacts.py (LaTeX tables & vector PDF generator)
├─ Execute paper export to populate reports/paper/tables/ (Tables 1, 2, 3, 4)
├─ Generate 300-DPI vector figures in reports/paper/figures/ (Figures 1, 2, 3, 4)
└─ Compile reports/paper/PAPER_RESULTS_DIGEST.md with p-values and percentage lifts

Day 7: Paper Drafting, Code Freeze & Deliverable Package
├─ Draft research paper sections using generated LaTeX tables and digest numbers
├─ Verify all unit tests pass (pytest tests/)
├─ Complete interactive demonstration notebooks (01_eda, 02_replication, 03_recourse)
└─ Final packaging, README documentation polish, and deliverable verification
```

---

## 11. Risk Management, Governance & Compliance Auditing

### 11.1 Technical & Computational Risk Mitigations
1. **Computational Bottleneck on 1.3M Rows:**
   - *Mitigation:* Dual-mode execution architecture allows 100% of testing, debugging, and tuning to run in minutes on a stratified 200,000-loan cohort before firing full benchmark runs.
2. **Gradient Instability in Custom Objectives:**
   - *Mitigation:* Automated finite-difference gradient tests (`tests/test_gradients.py`) verify analytical loss derivatives against numerical approximations, with second-order Hessians clipped at $\epsilon = 10^{-6}$ to prevent division by zero.
3. **Probability Miscalibration under Asymmetric Loss:**
   - *Mitigation:* Mandatory post-training isotonic regression calibration ensures thresholding operates on true posterior probabilities.

### 11.2 Fair Lending Compliance & Disparate Impact (ECOA Regulation B)
- **Protected Attribute Exclusion:** Features representing protected demographic characteristics (race, gender, marital status, religion) are strictly excluded from all training feature matrices.
- **Proxy Discrimination Audit:** Statistical correlation tests evaluate whether engineered financial ratios inadvertently correlate with excluded demographic proxies.
- **Adverse Action Fidelity:** Programmatic adverse action disclosures ensure that every rejected applicant receives mathematically verifiable, non-arbitrary reasons alongside actionable recourse paths.
