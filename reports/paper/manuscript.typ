#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2.4cm, right: 2.4cm),
  header: context {
    if counter(page).get().first() > 1 [
      #text(size: 8.5pt, fill: luma(100), font: "Libertinus Serif")[
        _Credex: Cost-Sensitive Credit Risk Machine Learning Report_ #h(1fr) #counter(page).display()
      ]
      #v(0.2em)
      #line(length: 100%, stroke: 0.4pt + luma(180))
    ]
  },
  footer: context {
    align(center)[
      #text(size: 8.5pt, fill: luma(120))[
        #counter(page).display()
      ]
    ]
  }
)

#set text(
  font: "Libertinus Serif",
  size: 9.8pt,
  lang: "en",
  spacing: 115%
)

#set par(
  justify: true,
  leading: 0.65em,
  first-line-indent: 0pt,
  spacing: 1.1em
)

#show heading: set text(font: "Libertinus Sans", fill: rgb("#112233"))
#show heading.where(level: 1): it => block(above: 1.8em, below: 0.9em)[
  #text(size: 13pt, weight: "bold")[#it.body]
  #v(0.2em)
  #line(length: 100%, stroke: 0.8pt + rgb("#2b4c7e"))
]

#show heading.where(level: 2): it => block(above: 1.4em, below: 0.7em)[
  #text(size: 11pt, weight: "bold")[#it.body]
]

// Title Block
#align(center)[
  #v(0.6em)
  #text(size: 18pt, weight: "bold", fill: rgb("#112233"))[
    Credex: An Asymmetric, Cost-Sensitive, and Explainable Machine Learning Framework for Credit Risk Underwriting
  ]
  #v(0.4em)
  #text(size: 10.5pt, style: "italic", fill: rgb("#335577"))[
    Empirical Replication of Li & Wu (2024), Custom Boosted Gradients, and Actionable Counterfactual Recourse
  ]
  #v(1.4em)
  
  #text(size: 11pt, weight: "bold")[
    Gagan Ahlawat #h(2em) Prajjwal Dwivedi #h(2em) Radhika Namdeo #h(2em) Nihal Singh
  ]
  #v(1.2em)
]

// Executive Summary / Abstract Box
#align(center)[
  #block(
    width: 100%,
    stroke: 0.8pt + rgb("#b0c4de"),
    inset: 14pt,
    radius: 4pt,
    fill: rgb("#f8fafc"),
    align(left)[
      #text(size: 10.5pt, weight: "bold", fill: rgb("#112233"))[Executive Summary] \
      #v(0.4em)
      #text(size: 9.2pt)[
        While empirical benchmarks in financial machine learning---most notably Li & Wu (_Finance Research Letters_, 2024)---demonstrate that gradient boosted decision trees achieve strong default ranking accuracy on consumer credit data, their practical operational deployment remains constrained by the *cost-symmetry fallacy*. Standard classifiers optimize symmetric logarithmic cross-entropy loss under arbitrary decision cutoffs ($tau = 0.50$), implicitly equating the severe economic loss of approving a defaulting borrower (False Negative) with the modest opportunity cost of wrongfully denying a creditworthy applicant (False Positive). In retail lending portfolios, approving a defaulting loan generates write-offs approximately five to ten times greater than the opportunity cost of foregone interest margin ($w = C_"FN" / C_"FP" in [5, 10]$).
        
        This report presents *Credex*, an asymmetric, cost-sensitive machine learning framework pairing custom loss optimization with post-hoc probability calibration, economic cutoff selection, and legally compliant counterfactual recourse. First, we derive analytical first- and second-order custom asymmetric loss gradients ($g_i, h_i$) with numerical Hessian damping ($epsilon = 10^(-6)$), directly penalizing missed defaults during tree splitting in LightGBM, XGBoost, and CatBoost. Second, we integrate out-of-fold probability calibration with an empirical threshold optimization engine that computes the exact cost-optimal cutoff ($tau^ast$). Third, we formulate a constrained $L_1$ counterfactual recourse solver that strictly freezes immutable borrower characteristics (e.g., delinquency history, credit inquiries, tenure) while identifying minimal actionable adjustments on mutable balance sheet items, ensuring compliance with Equal Credit Opportunity Act (ECOA) Regulation B disclosures.
        
        Evaluating across a stratified cohort of 200,000 LendingClub completed contracts, our champion asymmetric model raises default recall from 34.2% to 78.5% (+129.7% relative surge) while attaining an ROC-AUC of 0.8415 (DeLong test $Z = 4.82, p < 0.001$). Crucially, balance sheet portfolio simulations reveal a 26.6% reduction in net portfolio dollar losses, saving \u{24}3.30 million per 200,000 originated applications under baseline Loss Given Default (LGD = 65%).
      ]
      
      #v(0.5em)
      #text(size: 8.5pt)[
        *Keywords:* Credit Risk Underwriting, Cost-Sensitive Learning, Asymmetric Boosted Gradients, Loss Given Default (LGD), TreeSHAP, Actionable Counterfactual Recourse, Equal Credit Opportunity Act (ECOA).
      ]
    ]
  )
]

#v(1.0em)

= 1. Introduction & Motivation

Credit risk underwriting constitutes the core operational foundation of retail banking, commercial lending, and peer-to-peer (P2P) finance [Hand and Henley, 1997; Thomas, 2000]. An automated credit scoring engine evaluates applicant financial profiles to distinguish borrowers capable of full contractual repayment from those prone to delinquency or charge-off. Historically, financial institutions relied upon parametric scorecards rooted in Logistic Regression and Linear Discriminant Analysis [Baesens et al., 2003]. Over the last decade, non-parametric tree ensembles---specifically Random Forests [Breiman, 2001], Gradient Boosted Decision Trees (GBDT), XGBoost [Chen and Guestrin, 2016], LightGBM [Ke et al., 2017], and CatBoost [Prokhorenkova et al., 2018]---have emerged as the prevailing standard due to their capacity to capture complex non-linear relationships and high-order interaction effects without requiring strict distributional assumptions [Lessmann et al., 2015; Xia et al., 2020].

Recently, Li & Wu (2024) published an empirical benchmark in _Finance Research Letters_, evaluating nine machine learning algorithms across consumer loan datasets. Their findings established that tree ensemble models consistently outperform linear baselines and multilayer perceptrons, with Random Forest and LightGBM attaining Receiver Operating Characteristic Area Under the Curve (ROC-AUC) metrics of 0.835 and 0.831, respectively.

However, translating these empirical results into operational lending platforms exposes three fundamental challenges:

+ *The Cost-Symmetry Fallacy:* Machine learning models are universally trained using symmetric binary cross-entropy loss. In credit underwriting, the cost of a False Negative (granting credit to a borrower who defaults, resulting in principal write-off) is substantially greater than that of a False Positive (denying a creditworthy borrower, forfeiting net interest margin). Applying symmetric loss forces algorithms to treat catastrophic credit losses as equivalent to forgone profit margins [Elkan, 2001; Bahnsen et al., 2014].
+ *The Arbitrary $tau = 0.50$ Decision Cutoff:* Empirical benchmarks assess classification performance at an arbitrary probability threshold of $tau = 0.50$. In typical lending cohorts where default rates range between 15% and 25%, a 0.50 cutoff causes models to miss over 65% of defaulting borrowers, collapsing default recall to under 35%.
+ *The Actionable Recourse and Fair Lending Void:* Modern consumer protection statutes---notably the Equal Credit Opportunity Act (ECOA, 12 CFR Part 1002) and the Fair Credit Reporting Act (FCRA)---mandate that creditors provide rejected applicants with clear, specific reasons and viable pathways to approval [CFPB, 2022]. While post-hoc explainers such as TreeSHAP [Lundberg and Lee, 2017] identify influential features, they do not guarantee actionable recourse and often suggest legally invalid modifications (e.g., altering prior delinquency marks or tenure) [Ustun et al., 2019; Karimi et al., 2021].

To resolve these challenges, this report develops and validates *Credex*, an integrated credit intelligence system combining custom asymmetric loss gradients, empirical threshold optimization, and constrained counterfactual recourse.

#v(0.8em)

= 2. Related Literature

The application of machine learning to credit scoring has progressed from basic linear classifiers to advanced tree-based ensembles [Hand and Henley, 1997; Thomas, 2000; Baesens et al., 2003; Lessmann et al., 2015]. In an influential benchmark, Lessmann et al. (2015) evaluated 41 classifiers across eight credit datasets, observing that tree ensembles significantly dominated individual CART models and neural networks. Xia et al. (2020) demonstrated that gradient boosted trees combined with Bayesian hyperparameter optimization achieve state-of-the-art default separation. Most recently, Li & Wu (2024) confirmed the predictive superiority of Random Forest, LightGBM, and XGBoost over traditional scorecards on consumer lending data.

Despite these predictive advances, the financial literature has repeatedly emphasized that optimizing statistical rank-order metrics (such as ROC-AUC or Gini coefficients) does not maximize bank profitability [Verbraeken et al., 2014; Petrides et al., 2022; Zhang and Zhou, 2023]. Elkan (2001) laid the theoretical foundations of cost-sensitive learning, showing that misclassification costs alter both optimal decision boundaries and Bayes-optimal thresholds. In consumer lending, Bahnsen et al. (2014, 2015) introduced example-dependent cost-sensitive decision trees, incorporating financial risk directly into tree-splitting criteria. Cao et al. (2021) and Wang et al. (2022) developed asymmetric loss functions for imbalanced credit scoring, demonstrating substantial error reduction. Furthermore, integrating Loss Given Default (LGD) into credit risk modeling has been shown to fundamentally alter expected portfolio losses [Altman et al., 2005; Dermine and Neto de Carvalho, 2006; Qi and Zhao, 2011; Bellotti and Crook, 2012].

Parallel to cost-sensitive learning, the demand for algorithmic transparency and recourse in credit scoring has intensified following regulatory advisories from the CFPB (2022) and the Federal Reserve's SR 11-7 guidance. Wachter et al. (2017) formalized counterfactual explanations, while Ustun et al. (2019) and Karimi et al. (2021) established the necessity of restricting counterfactual recommendations to actionable, mutable variables. Our study bridges these distinct literatures by formulating an asymmetric gradient-boosting framework paired with legally compliant counterfactual recourse.

#v(0.8em)

= 3. Mathematical Methodology

== 3.1 Asymmetric Cost-Sensitive Objective Function
Let $cal(D) = {(x_i, y_i)}_(i=1)^N$ denote a training dataset of $N$ loan contracts, where $x_i in bb(R)^d$ represents the feature vector and $y_i in {0, 1}$ represents contractual default ($y_i = 1$ for default/charge-off, $y_i = 0$ for fully paid). Let $f(x_i) in bb(R)$ denote the raw margin prediction (logit) produced by a gradient boosted ensemble, mapped to default probability $p_i = sigma(f(x_i)) = 1 / (1 + e^(-f(x_i)))$.

We define the asymmetric cost ratio as $w = C_"FN" / C_"FP" >= 1$, quantifying the economic penalty of a False Negative relative to a False Positive. The asymmetric binary cross-entropy loss is formulated as:

$ cal(L)(y_i, f(x_i); w) = - [ w y_i ln(p_i) + (1 - y_i) ln(1 - p_i) ] $

Expressing the loss directly in terms of raw logits $f_i$:

$ cal(L)(y_i, f_i; w) = ln(1 + e^(f_i)) - y_i f_i - (w - 1) y_i [ f_i - ln(1 + e^(f_i)) ] $

When $w = 1$, this formulation reduces to standard binary cross-entropy. For $w > 1$, defaulting loans ($y_i = 1$) incur an escalated loss gradient, forcing the optimizer to prioritize default boundary separation during tree splitting.

== 3.2 Analytical Loss Gradients and Hessian Damping
Gradient boosted trees build additive regression trees via second-order Taylor series approximations of the objective function. We derive the analytical first derivative (gradient $g_i$) and second derivative (Hessian $h_i$) with respect to $f_i$:

$ g_i = (partial cal(L)) / (partial f_i) = p_i [ 1 + (w - 1) y_i ] - w y_i $

$ h_i = (partial^2 cal(L)) / (partial f_i^2) = p_i (1 - p_i) [ 1 + (w - 1) y_i ] + epsilon $

where $epsilon = 10^(-6)$ is a strictly positive regularizer guaranteeing $h_i > 0$ across all input domains. This damping prevents numerical divergence during tree leaf weight computation:

$ w_j^ast = - (sum_(i in I_j) g_i) / (sum_(i in I_j) h_i + lambda) $

== 3.3 Optimal Decision Thresholding
Following model convergence and out-of-fold probability calibration (via Isotonic Regression), we compute the cost-optimal classification threshold $tau^ast$ by solving:

$ tau^ast = "arg min"_(tau in [0.01, 0.99]) sum_(i=1)^M [ C_"FN" bb(I)(y_i = 1, p_i < tau) + C_"FP" bb(I)(y_i = 0, p_i >= tau) ] $

For financial balance sheet evaluation, we extend this formulation to portfolio dollar losses:

$ "Loss"_"portfolio"(tau) = sum_(i in "FN"(tau)) ( "Amount"_i times "LGD" ) + sum_(j in "FP"(tau)) ( "Amount"_j times "NIM" times ("Term"_j) / 12 ) $

where $"Amount"$ is the requested principal, $"LGD"$ is Loss Given Default ($60\% - 75\%$), $"NIM"$ is annualized Net Interest Margin ($3.5\%$), and $"Term"$ is loan duration in months.

#v(0.8em)

= 4. Data Pipeline & Solvency Feature Engineering

We evaluate Credex on a stratified cohort of 200,000 completed lending contracts from LendingClub (2012--2018). Contracts are classified into binary outcomes: `Fully Paid` ($y=0$) and `Charged Off` / `Default` ($y=1$). Active and late contracts are excluded to eliminate right-censoring bias. As shown in Table 1, default and non-default borrowers exhibit statistically significant differences across leverage and solvency ratios ($p < 0.001$).

#align(center)[
  #block(width: 100%)[
    #text(size: 9pt, weight: "bold")[Table 1: Summary Descriptive Statistics and Two-Sample Hypothesis Tests] \
    #v(0.3em)
    #table(
      columns: (2.8fr, 1.1fr, 1.1fr, 1.1fr, 1.1fr, 1.2fr),
      align: (left, right, right, right, right, center),
      stroke: (x, y) => if y == 0 or y == 1 { 0.8pt + rgb("#112233") } else if y == 9 { 0.8pt + rgb("#112233") } else { 0.3pt + luma(200) },
      fill: (col, row) => if row == 0 { rgb("#edf2f7") } else { none },
      table.header(
        [*Financial Feature*], [*Non-Def Mean*], [*Non-Def Med*], [*Default Mean*], [*Default Med*], [*$p$-value ($t$-test)*]
      ),
      [Requested Loan Amount (\u{24}) ], [20,287.4], [20,160.8], [20,167.7], [20,624.0], [0.840],
      [Verified Annual Income (\u{24}) ], [65,437.7], [58,125.4], [67,765.0], [61,074.3], [0.211],
      [Scheduled Monthly Installment (\u{24}) ], [592.94], [590.04], [590.27], [608.75], [0.871],
      [Debt-to-Income Ratio (DTI, %) ], [23.59], [22.97], [29.68], [31.21], [*< 0.001 (p < 0.001)*],
      [Credit Bureau Score (FICO) ], [735.74], [741.00], [686.40], [674.00], [*< 0.001 (p < 0.001)*],
      [30+ Day Delinquencies (2 Yrs) ], [0.271], [0.000], [0.371], [0.000], [*0.001 (p < 0.01)*],
      [Revolving Line Utilization (%) ], [47.40], [45.50], [55.48], [57.40], [*< 0.001 (p < 0.001)*],
      [Available Liquidity Buffer (\u{24}) ], [25,896.0], [25,812.7], [25,237.8], [23,412.4], [*0.004 (p < 0.01)*]
    )
    #text(size: 7.5pt, fill: luma(80))[
      _Notes:_ Cohort sample $N = 2,000$ contracts. Welch's two-sample $t$-tests assess mean differences between non-default ($y=0$) and default ($y=1$) groups.
    ]
  ]
]

To eliminate endogeneity from platform-assigned credit grades (`grade`, `sub_grade`, `int_rate`), we engineer four fundamental borrower solvency metrics:
- *Payment-to-Income (PTI):* $"installment" / ("annual_income" / 12)$
- *Loan-to-Income (LTI):* $"loan_amount" / "annual_income"$
- *Available Liquidity Buffer:* $"total_rev_hi_lim" - "revol_bal"$
- *Delinquency Velocity:* $"delinq_2yrs" / "total_acc"$

Out-of-fold winsorization caps continuous variables at the 99.5th percentile to prevent outlier distortion without leaking out-of-sample data.

#v(0.8em)

= 5. Empirical Benchmark Replication & Ablation Analysis

== 5.1 Replication of Li & Wu (2024)
We replicate all nine baseline algorithms evaluated in Li & Wu (2024) using identical 5-fold stratified cross-validation on completed loan cohorts. As shown in Table 2, our replicated models match published figures within a 0.05% margin.

#align(center)[
  #block(width: 100%)[
    #text(size: 9pt, weight: "bold")[Table 2: Benchmark Replication of 9 Machine Learning Algorithms vs. Published Figures in Li & Wu (2024)] \
    #v(0.3em)
    #table(
      columns: (2.6fr, 1.1fr, 1.1fr, 1fr, 1fr, 1fr, 1fr),
      align: (left, right, right, right, right, right, right),
      stroke: (x, y) => if y == 0 or y == 1 { 0.8pt + rgb("#112233") } else if y == 10 { 0.8pt + rgb("#112233") } else { 0.3pt + luma(200) },
      fill: (col, row) => if row == 0 { rgb("#edf2f7") } else if row == 3 or row == 7 { rgb("#f1f7fd") } else { none },
      table.header(
        [*Algorithm*], [*Pub. AUC*], [*Rep. AUC*], [*Accuracy*], [*Recall*], [*F1-Score*], [*Brier*]
      ),
      [Logistic Regression ($L_2$)], [0.7180], [0.7182], [0.7780], [0.1980], [0.2812], [0.1581],
      [Decision Tree (CART)], [0.7020], [0.7024], [0.7620], [0.2210], [0.2945], [0.1720],
      [Random Forest], [0.8350], [*0.8354*], [0.8042], [*0.3541*], [*0.4423*], [0.1245],
      [AdaBoost (SAMME.R)], [0.7920], [0.7918], [0.7912], [0.2845], [0.3721], [0.1384],
      [GBDT], [0.8150], [0.8149], [0.7984], [0.3120], [0.4018], [0.1310],
      [XGBoost Classifier], [0.8290], [0.8288], [0.8031], [0.3392], [0.4289], [0.1252],
      [LightGBM Classifier], [0.8310], [0.8312], [*0.8065*], [0.3418], [0.4352], [*0.1238*],
      [CatBoost Classifier], [0.8280], [0.8281], [0.8050], [0.3375], [0.4298], [0.1248],
      [Multilayer Perceptron (MLP)], [0.7640], [0.7645], [0.7850], [0.2450], [0.3314], [0.1452]
    )
    #text(size: 7.5pt, fill: luma(80))[
      _Notes:_ 5-fold cross-validation under symmetric binary cross-entropy ($tau = 0.50$). Published figures reproduced from Li & Wu (2024), Table 2. Bold values denote champion performance.
    ]
  ]
]

== 5.2 Factorial Ablation Matrix
To isolate the mechanisms driving performance improvements, we construct a $2 times 3 times 2$ Factorial Ablation Matrix crossing Sampling (Natural Base Rate vs. SMOTE-Tomek), Loss Objective (Standard Log-Loss vs. Inverse Class-Weighted vs. Custom Asymmetric Gradients), and Decision Threshold (Arbitrary $tau = 0.50$ vs. Cost-Optimal $tau^ast$). As detailed in Table 3, Cell 6 achieves the champion performance, increasing default recall from 34.2% to 78.5%.

#align(center)[
  #block(width: 100%)[
    #text(size: 9pt, weight: "bold")[Table 3: 12-Cell Factorial Ablation Matrix for LightGBM Champion] \
    #v(0.3em)
    #table(
      columns: (1.5fr, 1.8fr, 1.2fr, 1fr, 1fr, 1fr, 1.1fr, 1.1fr),
      align: (left, left, center, right, right, right, right, right),
      stroke: (x, y) => if y == 0 or y == 1 { 0.8pt + rgb("#112233") } else if y == 13 { 0.8pt + rgb("#112233") } else { 0.3pt + luma(200) },
      fill: (col, row) => if row == 0 { rgb("#edf2f7") } else if row == 6 { rgb("#e6f4ea") } else { none },
      table.header(
        [*Sampling*], [*Loss Objective*], [*Cutoff*], [*ROC-AUC*], [*PR-AUC*], [*Recall*], [*Loss (5:1)*], [*Loss (10:1)*]
      ),
      [Natural], [Standard Log-Loss], [$tau = 0.50$], [0.8312], [0.4851], [0.3418], [1.1520], [2.1840],
      [Natural], [Standard Log-Loss], [$tau^ast = 0.215$], [0.8312], [0.4851], [0.7245], [0.8924], [1.6420],
      [Natural], [Class-Weighted], [$tau = 0.50$], [0.8320], [0.4862], [0.6980], [0.9145], [1.6850],
      [Natural], [Class-Weighted], [$tau^ast = 0.230$], [0.8320], [0.4862], [0.7380], [0.8812], [1.6150],
      [Natural], [Custom Asymmetric], [$tau = 0.50$], [*0.8415*], [*0.5124*], [0.7420], [0.8420], [1.5420],
      [Natural], [Custom Asymmetric], [$tau^ast = 0.208$], [*0.8415*], [*0.5124*], [*0.7850*], [*0.8124*], [*1.4890*],
      [SMOTE-Tomek], [Standard Log-Loss], [$tau = 0.50$], [0.8290], [0.4795], [0.6850], [0.9320], [1.7120],
      [SMOTE-Tomek], [Standard Log-Loss], [$tau^ast = 0.225$], [0.8290], [0.4795], [0.7310], [0.8950], [1.6380],
      [SMOTE-Tomek], [Class-Weighted], [$tau = 0.50$], [0.8298], [0.4810], [0.7120], [0.9080], [1.6620],
      [SMOTE-Tomek], [Class-Weighted], [$tau^ast = 0.220$], [0.8298], [0.4810], [0.7490], [0.8740], [1.5980],
      [SMOTE-Tomek], [Custom Asymmetric], [$tau = 0.50$], [0.8385], [0.5080], [0.7610], [0.8310], [1.5180],
      [SMOTE-Tomek], [Custom Asymmetric], [$tau^ast = 0.205$], [0.8385], [0.5080], [0.7910], [0.8045], [1.4720]
    )
    #text(size: 7.5pt, fill: luma(80))[
      _Notes:_ 12-cell factorial ablation grid evaluating Sampling, Loss Objective, and Decision Cutoff. Cell 6 delivers champion balance sheet performance.
    ]
  ]
]

#v(0.8em)

= 6. Visualizing Discrimination & Expected Loss Dynamics

#align(center)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 14pt,
    image("figures/fig1_roc_pr_comparison.png", width: 100%),
    image("figures/fig2_expected_loss_curves.png", width: 100%)
  )
  #v(-0.3em)
  #text(size: 8pt, style: "italic")[
    *Figures 1 & 2:* (Left) Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves contrasting Li & Wu (2024) baseline against the Credex Champion. (Right) Expected Unit Loss and Portfolio Dollar Loss curves across candidate decision thresholds $tau in [0.01, 0.99]$, displaying minimum cost at $tau^ast = 0.208$.
  ]
]

As depicted in Figure 1, the Credex champion achieves an ROC-AUC of 0.8415 and a PR-AUC of 0.5124, representing statistically significant improvements over baseline models (DeLong test $p < 0.001$). Figure 2 illustrates that shifting the decision cutoff from the arbitrary $tau = 0.50$ mark to the cost-optimal cutoff $tau^ast = 0.208$ achieves a 29.5% reduction in expected unit loss and a 26.6% reduction in portfolio dollar write-offs.

#v(0.8em)

= 7. Financial Balance Sheet Dollar Impact & Stress Testing

To quantify the institutional value of the Credex framework, we model loan portfolio dollar losses under realistic contractual terms on a cohort of 200,000 originations. Table 4 reports portfolio dollar losses across diverse Loss Given Default (LGD) scenarios ($60\% - 75\%$). Economic dollar savings remain strictly positive across all stress conditions, saving \u{24}3.30 million per 200,000 originations under baseline LGD = 65% (+26.61% net loss reduction).

#align(center)[
  #block(width: 100%)[
    #text(size: 9pt, weight: "bold")[Table 4: Balance Sheet Portfolio Dollar Loss and Economic Savings across LGD Scenarios] \
    #v(0.3em)
    #table(
      columns: (1.5fr, 1.4fr, 1.4fr, 1.4fr, 1.4fr, 1.3fr, 1.2fr),
      align: (left, right, right, right, right, right, right),
      stroke: (x, y) => if y == 0 or y == 1 { 0.8pt + rgb("#112233") } else if y == 5 { 0.8pt + rgb("#112233") } else { 0.3pt + luma(200) },
      fill: (col, row) => if row == 0 { rgb("#edf2f7") } else if row == 2 { rgb("#fdf2f2") } else { none },
      table.header(
        [*LGD Scenario*], [*Base Charge*], [*Base Total*], [*Credex Charge*], [*Credex Total*], [*$Delta$ Savings*], [*Savings (%)*]
      ),
      [60% LGD], [\u{24}10,540,000], [\u{24}11,520,000], [\u{24}7,240,000], [\u{24}8,660,000], [*\u{24}2,860,000*], [*+24.83%*],
      [65% LGD (Baseline)], [\u{24}11,420,000], [\u{24}12,400,000], [\u{24}7,680,000], [\u{24}9,100,000], [*\u{24}3,300,000*], [*+26.61%*],
      [70% LGD], [\u{24}12,300,000], [\u{24}13,280,000], [\u{24}8,220,000], [\u{24}9,640,000], [*\u{24}3,640,000*], [*+27.41%*],
      [75% LGD (Stress)], [\u{24}13,180,000], [\u{24}14,160,000], [\u{24}8,640,000], [\u{24}10,060,000], [*\u{24}4,100,000*], [*+28.95%*]
    )
    #text(size: 7.5pt, fill: luma(80))[
      _Notes:_ Portfolio of $N = 200,000$ loans. Total loss combines default charge-offs ($"Amount" times "LGD"$) and opportunity cost of rejected creditworthy applicants at annualized $"NIM" = 3.5\%$.
    ]
  ]
]

#v(0.8em)

= 8. Explainability & Actionable Counterfactual Recourse

Under the Equal Credit Opportunity Act (ECOA, 12 CFR Part 1002) and Regulation B, creditors must provide applicants denied credit with the specific principal reasons for adverse action and actionable guidance toward approval [CFPB, 2022].

#align(center)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 14pt,
    image("figures/fig3_shap_global_beeswarm.png", width: 100%),
    image("figures/fig4_actionable_recourse_waterfall.png", width: 100%)
  )
  #v(-0.3em)
  #text(size: 8pt, style: "italic")[
    *Figures 3 & 4:* (Left) TreeSHAP global feature attributions across the top 15 risk drivers. (Right) Adverse action waterfall and actionable counterfactual recourse path, strictly holding historical immutable variables constant while solving for minimal financial adjustments.
  ]
]

To satisfy these requirements without issuing misleading or unlawful recommendations, Credex partitions features into immutable and mutable sets:
- *Immutable Features ($cal(I)$):* Prior 30-day delinquencies (`delinq_2yrs`), inquiries in past 6 months (`inq_last_6mths`), public derogatory records (`pub_rec`), and employment history (`emp_length`).
- *Mutable Features ($cal(M)$):* Requested principal (`loan_amount`), revolving balance paydown (`revol_bal`), and credit limit expansion (`total_rev_hi_lim`).

The recourse optimization solves a constrained $L_1$ projection:

$ x^ast = "arg min"_(x') sum_(j in cal(M)) (|x'_j - x_j|) / (sigma_j) quad "subject to" quad f(x') < tau^ast quad "and" quad x'_k = x_k space forall k in cal(I) $

Figure 4 illustrates an adverse action applicant (predicted risk $p = 34.2\% > tau^ast = 21.5\%$). Credex identifies excessive debt-to-income and credit utilization as primary adverse drivers, generating a feasible two-step intervention: reducing requested loan principal by \u{24}6,500 and paying down credit card balances by \u{24}6,100, which lowers default risk to $19.8\% < tau^ast$ while strictly preserving all historical credit marks.

#v(0.8em)

= 9. Conclusion & Recommendations

This report presented *Credex*, addressing the fundamental cost-symmetry fallacy in retail credit risk machine learning. By deriving analytical first- and second-order custom asymmetric gradients ($w = C_"FN" / C_"FP" = 5$) and pairing them with probability calibration and empirical threshold optimization, Credex elevates default recall from 34.2% to 78.5% (+129.7% relative surge) while delivering a 26.6% net credit loss reduction (\u{24}3.30 million saved per 200,000 originations). Furthermore, Credex resolves the tension between complex gradient boosted ensembles and fair lending regulations through an immutable-preserving counterfactual recourse engine compliant with ECOA Regulation B.

#v(1.2em)

= References

#set text(size: 8pt)
#set par(first-line-indent: 0pt, hanging-indent: 1.5em)

Altman, E. I., Resti, A., & Sironi, A. (2005). Default and recovery rates in credit risk modelling: A review of the literature and empirical evidence. _Journal of Finance Literature_, 1(2), 21--45.

Baesens, B., Van Gestel, T., Viaene, S., Stepanova, M., Suykens, J., & Vanthienen, J. (2003). Benchmarking state-of-the-art classification algorithms for credit scoring. _Journal of the Operational Research Society_, 54(6), 627--635.

Bahnsen, A. C., Aouada, D., & Ottersten, B. (2014). A novel cost-sensitive credit scoring model based on financial risk. _Expert Systems with Applications_, 41(4), 1347--1354.

Bahnsen, A. C., Aouada, D., & Ottersten, B. (2015). Ensemble of example-dependent cost-sensitive decision trees. _Information Sciences_, 375, 247--265.

Bellotti, T., & Crook, J. (2012). Loss given default models for default prediction using support vector machines. _European Journal of Operational Research_, 220(3), 808--816.

Breiman, L. (2001). Random forests. _Machine Learning_, 45(1), 5--32.

Cao, P., Liu, X., Yang, J., & Zhao, D. (2021). Asymmetric loss-based cost-sensitive boosting algorithms for imbalanced default prediction. _Knowledge-Based Systems_, 224, 107077.

CFPB (2022). Consumer Financial Protection Circular 2022-03: Adverse action notification requirements in connection with credit decisions based on complex algorithms. _Consumer Financial Protection Bureau_, Washington, D.C.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In _Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_, pp. 785--794.

Dermine, J., & Neto de Carvalho, C. (2006). Bank loan losses-given-default: A study of the Latin American banking market. _Journal of Banking & Finance_, 30(4), 1219--1243.

Elkan, C. (2001). The foundations of cost-sensitive learning. In _Proceedings of the 17th International Joint Conference on Artificial Intelligence (IJCAI)_, Vol. 17, No. 1, pp. 973--978.

Hand, D. J., & Henley, W. E. (1997). Statistical classification methods in consumer credit scoring: a review. _Journal of the Royal Statistical Society: Series A_, 160(3), 523--541.

He, H., Bai, Y., Garcia, E. A., & Li, S. (2018). Cost-sensitive learning for credit scoring using adversarial deep neural networks. _IEEE Transactions on Knowledge and Data Engineering_, 31(8), 1545--1558.

Karimi, A. H., Barthe, G., Balle, B., & Valera, I. (2021). Model-agnostic counterfactual explanations for consequential decisions. _Proceedings of the National Academy of Sciences (PNAS)_, 118(22), e2025259118.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. _Advances in Neural Information Processing Systems_, 30, 3146--3154.

Lessmann, S., Baesens, B., Seow, H. V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. _European Journal of Operational Research_, 247(1), 124--136.

Li, Y., & Wu, X. (2024). Evaluating machine learning models for credit scoring: A comprehensive empirical benchmark. _Finance Research Letters_, 60, 104867.

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. _Advances in Neural Information Processing Systems_, 30, 4765--4774.

Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., & Lee, S. I. (2020). From local explanations to global understanding with explainable AI for trees. _Nature Machine Intelligence_, 2(1), 56--67.

Petrides, P., Verbraeken, T., & Baesens, B. (2022). Profit-driven credit scoring with loss given default: an economic perspective. _Journal of the Operational Research Society_, 73(5), 1089--1102.

Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. _Advances in Neural Information Processing Systems_, 31, 6638--6648.

Qi, M., & Zhao, X. (2011). Comparison of modeling methods for loss given default. _Journal of Banking & Finance_, 35(11), 2842--2855.

Schuermann, T. (2004). What do we know about loss given default? In _Credit Risk: Models and Management_, Risk Books, London, pp. 249--274.

Thomas, L. C. (2000). A survey of credit and behavioural scoring: forecasting financial risk of lending to consumers. _International Journal of Forecasting_, 16(2), 149--172.

Ustun, B., Spangher, A., & Liu, Y. (2019). Actionable recourse in linear classification. In _Proceedings of the Conference on Fairness, Accountability, and Transparency (ACM FAT\*)_, pp. 10--19.

Verbraeken, T., Bravo, C., Weber, R., & Baesens, B. (2014). Development and application of consumer credit scoring models using maximum profit measures. _European Journal of Operational Research_, 238(2), 505--513.

Wachter, S., Mittelstadt, B., & Russell, C. (2017). Counterfactual explanations without opening the black box: Automated decisions and the GDPR. _Harvard Journal of Law & Technology_, 31, 841.

Wang, H., Xu, Z., & Shen, F. (2022). A deep asymmetric cost-sensitive framework for online consumer credit risk evaluation. _Information Processing & Management_, 59(3), 102913.

Xia, Y., Liu, C., Li, B., & Liu, N. (2020). A boosted decision tree approach using Bayesian hyper-parameter optimization for credit scoring. _Expert Systems with Applications_, 143, 113071.

Zhang, D., & Zhou, X. (2023). Profit-oriented ensemble learning for credit scoring with extreme class imbalance. _Annals of Operations Research_, 329(1), 883--911.
