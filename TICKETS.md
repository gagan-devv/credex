# CREDEX Work Breakdown: Parallel Lanes & Tickets

This document defines the complete engineering tickets for the **Credex Platform**, structured across 7 parallel execution lanes. Each ticket defines end-to-end functionality, strict unit and property-based test criteria, and dependency edges.

---

## Lane Overview & Dependency Graph

```
┌────────────────────────────────────────────────────────┐
│ Lane A: Foundation, Packaging & Configs (CREDEX-01..03)│
└──────────────────────────┬─────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
┌───────────────────────────┐ ┌──────────────────────────────────────────────┐
│ Lane B: Data Pipeline &   │ │ Lane D: Custom Loss Gradients & Economic Loss│
│ Features (CREDEX-04..07)  │ │ Optimization Engine (CREDEX-10..12)          │
└────────────┬──────────────┘ └──────────────────────┬───────────────────────┘
             │                                       │
             ▼                                       │
┌───────────────────────────┐                        │
│ Lane C: Baseline 9 Models │                        │
│ Replication (CREDEX-08..09│                        │
└────────────┬──────────────┘                        │
             │                                       │
             ├───────────────────────────────────────┤
             ▼                                       ▼
┌───────────────────────────┐ ┌──────────────────────────────────────────────┐
│ Lane E: Imbalance &       │ │ Lane F: Explainability & Actionable Recourse │
│ Ablation (CREDEX-13..15)  │ │ Engine (CREDEX-16..17)                       │
└────────────┬──────────────┘ └──────────────────────┬───────────────────────┘
             │                                       │
             └───────────────────┬───────────────────┘
                                 ▼
┌────────────────────────────────────────────────────────┐
│ Lane G: Paper Artifacts & CLI Runner (CREDEX-18..19)   │
└────────────────────────────────────────────────────────┘
```

---

## Lane A: Core Foundation, Packaging & Configs

### CREDEX-01: Python Packaging, Dockerfile & UV Virtual Environment
- **Branch:** `lane-a/infra-packaging`
- **Blocked by:** None (can start immediately)
- **What to build:** Complete Python packaging with `pyproject.toml`, pinned `requirements.txt`, `setup.py`, `Makefile`, containerized `Dockerfile`, and automated `uv` virtual environment setup.
- **Deliverables:**
  - `pyproject.toml`: Modern PEP 621 package config with metadata, dependencies, and tool settings (`pytest`, `black`, `ruff`).
  - `requirements.txt`: Strictly pinned production and testing dependencies.
  - `setup.py`: Setuptools wrapper for editable local installs (`pip install -e .`).
  - `Dockerfile`: Multi-stage Python 3.10+ Linux container with `uv` for reproducible execution on cloud/Kaggle compute.
  - `Makefile`: Targets for `install`, `lint`, `test`, `docker-build`, `clean`.
- **Acceptance Criteria:**
  - [x] `uv venv` creates clean virtual environment and installs all dependencies in $< 15$ seconds.
  - [x] `pip install -e .` succeeds in Python 3.10+.
  - [x] `docker build -t credex:latest .` successfully builds image.
  - [x] Unit test: Verify package importability `import src; assert src.__version__`.

### CREDEX-02: Base Config Schemas & Pydantic Validation
- **Branch:** `lane-a/config-schemas`
- **Blocked by:** CREDEX-01
- **What to build:** Centralized configuration schemas with strict runtime type validation using Pydantic and PyYAML.
- **Deliverables:**
  - `configs/base_config.yaml`: Random seeds, data paths, sample sizes, train/test OOT date boundaries.
  - `configs/features_config.yaml`: Baseline vs. enhanced feature lists, categorical/numerical registries, pricing feature toggle.
  - `configs/models_config.yaml`: Default hyperparameters for all 9 baseline algorithms and gradient boosters.
  - `configs/cost_matrix.yaml`: Penalty ratios ($5:1, 10:1$), LGD ($0.60–0.75$), NIM ($0.035$).
  - `src/config.py`: Pydantic V2 models (`BaseConfig`, `FeatureConfig`, `ModelConfig`, `CostMatrixConfig`) with YAML loaders.
- **Acceptance Criteria:**
  - [x] Invalid YAML values (e.g. negative costs, invalid ratios) raise descriptive Pydantic validation errors.
  - [x] Unit & Property tests: Validate that configuration loader parses defaults, overrides, and environment variables idempotently.

### CREDEX-03: Testing Harness & Property-Based Test Infrastructure
- **Branch:** `lane-a/test-harness`
- **Blocked by:** CREDEX-01
- **What to build:** Pytest configuration with Hypothesis property-based testing fixtures, synthetic financial tabular data generators, and test runners.
- **Deliverables:**
  - `tests/conftest.py`: Reusable fixtures for synthetic credit data (LendingClub schema and South German Credit schema).
  - Hypothesis custom strategies for generating realistic credit applicant profiles (`dti`, `annual_income`, `loan_amount`, `delinq_2yrs`).
- **Acceptance Criteria:**
  - [x] `pytest tests/` runs cleanly with zero failures.
  - [x] Property test: Generated synthetic datasets satisfy economic invariants (income $> 0$, utilization $\in [0, 100\%]$).

---

## Lane B: Data Pipeline & Feature Engineering

### CREDEX-04: Data Ingestion Engine (LendingClub & South German Credit)
- **Branch:** `lane-b/data-loader`
- **Blocked by:** CREDEX-02
- **What to build:** Ingestion pipeline supporting both the LendingClub historical completed loans dataset and the UCI South German Credit dataset.
- **Deliverables:**
  - `src/data/loader.py`: `load_lending_club()` and `load_german_credit()` with automatic schema casting and caching.
  - Dual-mode support: Deterministic stratified sampling of 200,000 completed loans (for fast tuning) vs. full 1.3M+ dataset (`--full-dataset`).
  - Target filtering: Isolates completed contracts (`loan_status` $\in$ {`Fully Paid` $\rightarrow 0$, `Charged Off` / `Default` $\rightarrow 1$}), rejecting active/late loans.
- **Acceptance Criteria:**
  - [x] Loads and validates raw CSV / parquet inputs against schema.
  - [x] Unit tests: Validate correct binary target mapping and sample size stratification.
  - [x] Property test: Target class distribution in sample matches population within $\pm 0.5\%$.

### CREDEX-05: Out-of-Fold Preprocessing & Financial Winsorization
- **Branch:** `lane-b/data-preprocessor`
- **Blocked by:** CREDEX-04
- **What to build:** Leakage-free preprocessing pipeline handling missing value imputation, winsorization, standard scaling, and categorical encoding.
- **Deliverables:**
  - `src/data/preprocessor.py`: `CreditDataPreprocessor` fitted strictly on training folds and transformed onto test folds.
  - Financial winsorization at the 99.5th percentile for extreme values (`annual_income`, `revol_bal`, `dti`).
  - Categorical encoders (One-Hot / Target encoding with out-of-fold regularization).
- **Acceptance Criteria:**
  - [x] No test data leakage during fit/transform operations.
  - [x] Winsorization caps outliers at the exact training 99.5th percentile without modifying normal values.
  - [x] Unit & Property tests: Test transform output dimensions, no NaN values remaining, and bounded variances.

### CREDEX-06: Dual Feature Regimes & Endogeneity Isolation
- **Branch:** `lane-b/feature-engineering`
- **Blocked by:** CREDEX-05
- **What to build:** Feature engineering engine delivering `feature_set_baseline` (~15 features reproducing Li & Wu 2024) and `feature_set_enhanced` (novel domain solvency ratios with pricing ablation).
- **Deliverables:**
  - `src/data/feature_engineering.py`:
    - Payment-to-Income (PTI): $\text{installment} / (\text{annual\_income} / 12)$
    - Loan-to-Income (LTI): $\text{loan\_amount} / \text{annual\_income}$
    - Available Liquidity Buffer: $\text{total\_rev\_hi\_lim} - \text{revol\_bal}$
    - Delinquency Velocity: $\text{delinq\_2yrs} / \text{total\_acc}$
    - Toggle `include_pricing_features: false` purging `interest_rate`, `grade`, `sub_grade`.
- **Acceptance Criteria:**
  - [x] Correct mathematical calculation of all 4 solvency ratios with division-by-zero guards.
  - [x] Unit & Property tests: Assert that disabling pricing features removes all lender-assigned grading columns.

### CREDEX-07: Partitioning Engine & Target Leakage Prevention
- **Branch:** `lane-b/partition-leakage`
- **Blocked by:** CREDEX-06
- **What to build:** Partitioning engine generating Stratified 5-Fold Cross-Validation splits and Out-of-Time (OOT) holdouts, with programmatic leakage audits.
- **Deliverables:**
  - Partitioning module splitting development cohort (2012–2016) and OOT validation holdout (2017–2018).
  - `tests/test_data_leakage.py`: Comprehensive test verifying zero post-origination columns (`recoveries`, `collection_recovery_fee`, `last_pymnt_amnt`) appear in predictor matrix.
- **Acceptance Criteria:**
  - [x] Stratified 5-fold splits maintain identical default rates across all folds.
  - [x] OOT split contains zero temporal overlap between train and test dates.
  - [x] Automated leakage unit tests pass with 100% assertions satisfied.

---

## Lane C: Benchmark Replication & Baseline Models

### CREDEX-08: Replication Suite for 9 Baseline Algorithms
- **Branch:** `lane-c/baseline-models`
- **Blocked by:** CREDEX-07
- **What to build:** Implementation and training wrappers for all 9 machine learning algorithms evaluated in Li & Wu (2024) under symmetric cross-entropy.
- **Deliverables:**
  - `src/models/baselines.py`: Unified interface for:
    1. Logistic Regression ($L_2$)
    2. Decision Tree Classifier
    3. Random Forest Classifier
    4. AdaBoost
    5. GBDT
    6. XGBoost
    7. LightGBM
    8. CatBoost
    9. Multilayer Perceptron (MLP)
- **Acceptance Criteria:**
  - [x] All 9 models train and predict probabilities via a standard scikit-learn compatible estimator API.
  - [x] Unit tests: Verify `fit()`, `predict_proba()`, and `predict()` across all 9 algorithms on synthetic and real sample splits.

### CREDEX-09: Probability Calibration & Table 2 Benchmark Reproduction
- **Branch:** `lane-c/calibration-benchmark`
- **Blocked by:** CREDEX-08
- **What to build:** Calibration module applying Isotonic Regression and Platt scaling, plus verification runner reproducing Table 2 of Li & Wu (2024).
- **Deliverables:**
  - `src/models/calibration.py`: Out-of-fold probability calibrator with Brier score evaluation.
  - Baseline verification script verifying Random Forest achieves ROC-AUC $\approx 0.835$ and LightGBM $\approx 0.831$.
- **Acceptance Criteria:**
  - [x] Calibrated probabilities monotonically align with empirical default frequencies.
  - [x] Baseline replication logs match published Table 2 metrics within expected empirical confidence bounds.
  - [x] Unit & Property tests: Assert calibrated probabilities remain strictly within $[0, 1]$.

---

## Lane D: Asymmetric Cost-Sensitive Learning & Threshold Optimization

### CREDEX-10: Analytical Custom Loss Gradients & Numerical Verification
- **Branch:** `lane-d/custom-gradients`
- **Blocked by:** CREDEX-02
- **What to build:** Exact mathematical implementation of custom first- and second-order loss derivatives ($g_i, h_i$) for asymmetric cost-sensitive binary cross-entropy ($w = C_{FN}/C_{FP} \ge 5$).
- **Deliverables:**
  - `src/models/objective_functions.py`:
    - First derivative: $g_i = p_i (1 + (w - 1) y_i) - w y_i$
    - Second derivative: $h_i = p_i (1 - p_i) (1 + (w - 1) y_i)$ with Hessian damping $\epsilon = 10^{-6}$.
  - `tests/test_gradients.py`: Finite-difference numerical gradient verification and Hypothesis property tests.
- **Acceptance Criteria:**
  - [x] Analytical gradients match numerical finite-difference approximations within tolerance $|g_{\text{analytical}} - g_{\text{numerical}}| < 10^{-5}$.
  - [x] Property test: For $w=1$, gradients exactly match standard binary cross-entropy loss derivatives.
  - [x] Second-order Hessians are strictly positive ($h_i > 0$) across all inputs, preventing tree optimizer divergence.

### CREDEX-11: Cost-Sensitive Gradient Boosted Ensembles
- **Branch:** `lane-d/cost-sensitive-models`
- **Blocked by:** CREDEX-10
- **What to build:** Custom objective integration into LightGBM, CatBoost, and XGBoost wrappers to penalize False Negatives at tree-split time.
- **Deliverables:**
  - `src/models/cost_sensitive.py`: Cost-sensitive wrappers injecting custom objective functions into LightGBM (`fobj`), XGBoost (`objective`), and CatBoost.
- **Acceptance Criteria:**
  - [x] Tree models train smoothly with custom objectives without NaN loss or gradient explosion.
  - [x] Unit tests: Verify cost-sensitive models produce higher recall on default cases compared to standard log-loss models.

### CREDEX-12: Economic Loss Threshold Optimizer (Unit & Portfolio Dollar Loss)
- **Branch:** `lane-d/threshold-optimizer`
- **Blocked by:** CREDEX-11
- **What to build:** Optimization engine sweeping decision thresholds $\tau \in [0.01, 0.99]$ to find $\arg\min \mathbb{E}[\text{Loss}(\tau)]$ under unit cost matrices ($5:1, 10:1$) and portfolio dollar losses (LGD $\times$ loan amount).
- **Deliverables:**
  - `src/models/threshold_optimizer.py`: Vectorized expected loss calculation and optimal $\tau^*$ identification.
  - `tests/test_threshold_optimizer.py`: Unit and property tests verifying loss convex-curve minimums.
- **Acceptance Criteria:**
  - [x] Optimal $\tau^*$ under $C_{FN}:C_{FP} = 5:1$ is strictly less than 0.50 (typically $\tau^* \in [0.15, 0.30]$).
  - [x] Property test: As $w = C_{FN}/C_{FP}$ increases, the optimal threshold $\tau^*$ strictly monotonically decreases (higher risk aversion).

---

## Lane E: Imbalance Remediation & Factorial Ablation Matrix

### CREDEX-13: SMOTE-Tomek Hybrid Resampling Engine
- **Branch:** `lane-e/imbalance-resampler`
- **Blocked by:** CREDEX-07
- **What to build:** Imbalance remediation pipeline combining synthetic minority oversampling (SMOTE) with Tomek link boundary noise elimination.
- **Deliverables:**
  - `src/data/resampler.py`: Safe resampling pipeline applied strictly to training folds.
- **Acceptance Criteria:**
  - [x] Resampling balances minority default class while Tomek links eliminate borderline ambiguous instances.
  - [x] Unit tests: Verify test partitions remain completely unresampled (preserving true empirical base rates).

### CREDEX-14: Comprehensive Financial & Statistical Evaluation Suite
- **Branch:** `lane-e/metrics-suite`
- **Blocked by:** CREDEX-12
- **What to build:** Full evaluation metrics suite computing both statistical discrimination and financial balance-sheet impact.
- **Deliverables:**
  - `src/evaluation/metrics.py`: Computes ROC-AUC, PR-AUC, Recall@Default, F1-Score, Brier Score, Expected Unit Loss ($5:1, 10:1$), and Total Portfolio Dollar Loss ($\Delta \$$ and $\%$ savings vs. Li & Wu).
- **Acceptance Criteria:**
  - [x] Calculates all statistical and financial metrics in a single pass.
  - [x] Unit & Property tests: Brier score strictly $\in [0, 1]$; dollar loss reductions match theoretical savings.

### CREDEX-15: Factorial Ablation Matrix Runner ($2 \times 3 \times 2$)
- **Branch:** `lane-e/ablation-runner`
- **Blocked by:** CREDEX-13, CREDEX-14
- **What to build:** Automated execution engine evaluating all 12 combinations of Sampling $\times$ Objective Loss $\times$ Decision Cutoff across 5-fold CV and OOT holdout.
- **Deliverables:**
  - `src/evaluation/ablation_runner.py`: Orchestrates multi-model factorial ablation and saves structured JSON/CSV run artifacts in `experiments/runs/`.
- **Acceptance Criteria:**
  - [x] Executes full 12-cell matrix across champion tree models (LightGBM, CatBoost, XGBoost, Random Forest).
  - [x] Outputs structured run records with mean and standard deviation across cross-validation folds.

---

## Lane F: Explainable AI & Constrained Counterfactual Recourse

### CREDEX-16: Fast TreeSHAP Attribution & Interaction Engine
- **Branch:** `lane-f/shap-explainer`
- **Blocked by:** CREDEX-09, CREDEX-11
- **What to build:** High-speed TreeSHAP attribution engine generating global beeswarm, feature dependence, and local waterfall plots.
- **Deliverables:**
  - `src/explainability/shap_explainer.py`: C++ optimized `TreeExplainer` attribution extraction ($N \le 5,000$).
  - `src/explainability/visualizer.py`: Matplotlib/Seaborn vector plot rendering (beeswarm, interaction, waterfall).
- **Acceptance Criteria:**
  - [x] TreeSHAP values satisfy additivity: $\phi_0 + \sum \phi_j(x) = f(x)$ within $10^{-4}$ numerical tolerance.
  - [x] High-resolution vector figures saved in `reports/paper/figures/`.

### CREDEX-17: Constrained Counterfactual Recourse Engine (ECOA Compliant)
- **Branch:** `lane-f/counterfactual-recourse`
- **Blocked by:** CREDEX-16, CREDEX-12
- **What to build:** Actionable recourse engine solving for minimal $L_1$ perturbations on mutable features (`loan_amount`, `revol_bal`) to flip decisions to approved ($p < \tau^*$), strictly holding immutable features (`delinq_2yrs`) constant.
- **Deliverables:**
  - `src/explainability/adverse_action.py`: Vectorized recourse projection solver ($<50$ ms per applicant) and automated Regulation B adverse action notice generator.
  - `tests/test_counterfactual_recourse.py`: Unit and property tests verifying immutable feature invariance.
- **Acceptance Criteria:**
  - [x] Solved counterfactuals guarantee $f(x_{\text{recourse}}) < \tau^*$.
  - [x] Property test: Immutable features (`delinq_2yrs`, `emp_length`) remain strictly identical between original and counterfactual applicant profiles.
  - [x] Generates standardized adverse action disclosure notice text compliant with ECOA / FCRA.

---

## Lane G: Automated Paper Artifacts & CLI Orchestration

### CREDEX-18: Automated Paper Artifact Generator (LaTeX & 300-DPI Vector Figures)
- **Branch:** `lane-g/paper-artifacts`
- **Blocked by:** CREDEX-15, CREDEX-17
- **What to build:** Exporter compiling empirical outputs directly into publication-ready LaTeX tables, 300-DPI vector figures, and executive results digest.
- **Deliverables:**
  - `src/evaluation/paper_artifacts.py`:
    - `table1_descriptive_stats.tex` (Feature distributions, $t$-tests)
    - `table2_li_wu_replication.tex` (9-model replication vs. published Table 2)
    - `table3_cost_sensitive_ablation.tex` ($2 \times 3 \times 2$ factorial ablation matrix)
    - `table4_economic_dollar_savings.tex` (Portfolio dollar savings and LGD sensitivity)
    - Vector figures: `fig1_roc_pr_comparison.pdf`, `fig2_expected_loss_curves.pdf`, `fig3_shap_global_beeswarm.pdf`, `fig4_actionable_recourse_waterfall.pdf`
    - `reports/paper/PAPER_RESULTS_DIGEST.md`: Executive summary with metrics, percentage lifts, and $p$-values.
- **Acceptance Criteria:**
  - [x] Valid LaTeX tables compile cleanly without syntax errors in standard document classes.
  - [x] All PDF figures conform to 300+ DPI vector standards with legible typography.

### CREDEX-19: Centralized Modular CLI & One-Shot Pipeline Orchestrator
- **Branch:** `lane-g/cli-orchestrator`
- **Blocked by:** CREDEX-01 through CREDEX-18
- **What to build:** Modular CLI entrypoint coordinating independent stage execution and end-to-end master runs.
- **Deliverables:**
  - `src/cli.py`: Subcommands for `ingest`, `replicate-paper`, `run-ablation`, `explain`, `export-paper`, and `run-all`.
- **Acceptance Criteria:**
  - [x] Each CLI subcommand runs independently with configurable flags and robust error handling.
  - [x] `python -m src.cli run-all --sample-size 200000` executes full pipeline end-to-end and exits with code 0.
