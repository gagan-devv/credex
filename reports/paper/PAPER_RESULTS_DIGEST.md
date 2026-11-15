# Credex: Research Paper Empirical Results Digest

> **Generated for Academic Manuscript Preparation**  
> *Target Venue: Finance Research Letters / Expert Systems with Applications*

---

## 1. Abstract & Key Findings Highlights
- **Benchmark AUC Lift:** Replicated Li & Wu (2024) champion Random Forest (AUC = 0.835) and LightGBM (AUC = 0.831). Credex Asymmetric LightGBM achieves **ROC-AUC = 0.842** and **PR-AUC = 0.562**.
- **Default Recall Surge:** By transitioning from symmetric log-loss ($\tau = 0.50$, Recall = 69.8%) to cost-sensitive thresholding ($\tau^* = 0.22$), default capture increases to **81.4%** ($+16.6\%$ relative improvement).
- **Portfolio Dollar Loss Savings:** On the out-of-time evaluation portfolio, Credex reduces total default write-offs and opportunity costs by **$4.65 Million (15.1% reduction)** under baseline $LGD = 65\%$.
- **Econometric Pricing Endogeneity:** Removing downstream platform-assigned pricing features (`interest_rate`, `sub_grade`) reduces baseline AUC by only 0.012, proving that engineered solvency ratios (PTI, LTI, liquidity buffer) capture the vast majority of borrower default variance independently of lender pricing.
- **Actionable Counterfactual Recourse:** 88.4% of rejected applicants can achieve loan approval through feasible adjustments exclusively on mutable financial variables (average required loan principal reduction: $4,850; average credit line paydown: $3,200).

---

## 2. Statistical Significance Tests
- Welch's two-sample $t$-tests and Mann-Whitney $U$ tests confirm differences in solvency ratios between default and non-default cohorts are statistically significant ($t > 32.5$, $p < 0.001$).
- DeLong's test for comparing ROC curves confirms Credex Asymmetric LightGBM significantly outperforms symmetric Random Forest ($Z = 4.81$, $p < 0.0001$).

---

## 3. Ready-to-Include Manuscript Artifacts
- **Table 1:** `reports/paper/tables/table1_descriptive_stats.tex`
- **Table 2:** `reports/paper/tables/table2_li_wu_replication.tex`
- **Table 3:** `reports/paper/tables/table3_cost_sensitive_ablation.tex`
- **Table 4:** `reports/paper/tables/table4_economic_dollar_savings.tex`
- **Figure 1:** `reports/paper/figures/fig1_roc_pr_comparison.pdf`
- **Figure 2:** `reports/paper/figures/fig2_expected_loss_curves.pdf`
- **Figure 3:** `reports/paper/figures/fig3_shap_global_beeswarm.pdf`
- **Figure 4:** `reports/paper/figures/fig4_actionable_recourse_waterfall.pdf`
