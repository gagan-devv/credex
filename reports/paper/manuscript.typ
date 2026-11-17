#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2.2cm, right: 2.2cm),
  header: context {
    if counter(page).get().first() > 1 [
      #align(center)[
        #text(size: 8.5pt, fill: luma(100), font: "DejaVu Serif")[
          _Credex: An Asymmetric, Cost-Sensitive Credit Risk Architecture_ #h(1fr) #counter(page).display()
        ]
      ]
    ]
  },
  footer: context {
    if counter(page).get().first() == 1 [
      #align(center)[
        #text(size: 8pt, fill: luma(120))[
          *Target Venue:* _Finance Research Letters_ / _Expert Systems with Applications_ (Working Paper Series)
        ]
      ]
    ]
  }
)

#set text(
  font: "DejaVu Serif",
  size: 10pt,
  lang: "en",
  spacing: 120%
)

#set par(
  justify: true,
  leading: 0.65em,
  first-line-indent: 1.2em
)

#show heading: set text(font: "DejaVu Sans", fill: rgb("#1a2a3a"))
#show heading.where(level: 1): it => block(above: 1.8em, below: 1.0em)[
  #text(size: 13pt, weight: "bold")[#it.body]
  #v(0.2em)
  #line(length: 100%, stroke: 0.8pt + rgb("#2b4c7e"))
]

#show heading.where(level: 2): it => block(above: 1.4em, below: 0.8em)[
  #text(size: 11pt, weight: "bold")[#it.body]
]

// Title Block
#align(center)[
  #v(0.5em)
  #text(size: 17pt, weight: "bold", fill: rgb("#0d233a"))[
    Credex: An Asymmetric, Cost-Sensitive, and Explainable Machine Learning Framework for Retail Credit Risk Underwriting
  ]
  #v(0.6em)
  #text(size: 10.5pt, style: "italic", fill: rgb("#335577"))[
    Extending Empirical Benchmarks with Custom Gradient Optimization, Economic Loss Minimization, and Regulatory Adverse Action Recourse
  ]
  #v(1.2em)
  
  #text(size: 11pt, weight: "bold")[Gagan Ahlawat]#super[1] #h(1.5em)
  #text(size: 11pt)[Credex Research Team]#super[1]
  
  #v(0.4em)
  #text(size: 9pt, fill: luma(80))[
    #super[1] Quantitative Finance & Intelligent Financial Systems Group \
    Repository & Research Artifacts: `https://github.com/gagan-devv/credex`
  ]
  #v(0.6em)
  #text(size: 8.5pt, fill: luma(100))[Date: November 2026]
  #v(1.0em)
]

// Abstract Box
#align(center)[
  #block(
    width: 95%,
    stroke: 0.8pt + rgb("#a0b2c6"),
    inset: 14pt,
    radius: 4pt,
    fill: rgb("#f8fafc"),
    align(left)[
      #text(size: 10.5pt, weight: "bold", fill: rgb("#1a2a3a"))[Abstract] \
      #v(0.4em)
      #text(size: 9.2pt)[
        While empirical benchmarks in financial machine learning---notably Li & Wu (_Finance Research Letters_, 2024)---demonstrate that gradient boosted decision trees achieve strong default ranking accuracy on consumer credit data, their operational deployment remains fundamentally impaired by the *cost-symmetry fallacy*. Standard classifiers optimize symmetric logarithmic cross-entropy loss under arbitrary decision cutoffs ($tau = 0.50$), implicitly equating the severe economic loss of approving a defaulting borrower (False Negative) with the modest opportunity cost of wrongfully denying a creditworthy applicant (False Positive). In retail lending, approving a defaulting loan generates write-offs approximately five to ten times greater than the opportunity cost of foregone interest margin ($w = C_"FN" / C_"FP" in [5, 10]$).
        
        In this paper, we present *Credex*, an asymmetric, cost-sensitive machine learning framework pairing custom loss optimization with post-hoc probability calibration, economic cutoff selection, and legally compliant counterfactual recourse. First, we derive analytical first- and second-order custom asymmetric loss gradients ($g_i, h_i$) with numerical Hessian damping ($epsilon = 10^(-6)$), directly penalizing missed defaults during tree splitting in LightGBM, XGBoost, and CatBoost. Second, we integrate out-of-fold probability calibration with an empirical threshold optimization engine that computes the exact cost-optimal cutoff ($tau^ast$). Third, we formulate a constrained $L_1$ counterfactual recourse solver that strictly freezes immutable borrower characteristics (e.g., delinquency history, credit inquiries, tenure) while identifying minimal actionable adjustments on mutable balance sheet items, ensuring compliance with Equal Credit Opportunity Act (ECOA) Regulation B disclosures.
        
        Evaluating across a stratified cohort of 200,000 LendingClub completed contracts, our champion asymmetric model raises default recall from 34.2% to 78.5% (+129.7% relative surge) while attaining an ROC-AUC of 0.8415 (DeLong test $Z = 4.82, p < 0.001$). Crucially, on a standard loan portfolio, Credex delivers a 26.6% reduction in net portfolio dollar losses, saving \$3.30 million per 200,000 originated applications under baseline Loss Given Default (LGD = 65%).
      ]
      
      #v(0.6em)
      #text(size: 8.8pt)[
        *Keywords:* Credit Risk Underwriting, Asymmetric Cost-Sensitive Learning, Custom Boosted Gradients, Loss Given Default (LGD), TreeSHAP, Actionable Counterfactual Recourse, Equal Credit Opportunity Act (ECOA).
      ]
    ]
  )
]

#v(1.2em)

= 1. Introduction & Background

Credit risk underwriting constitutes the foundation of commercial lending and modern peer-to-peer (P2P) finance. An automated credit scoring engine evaluates applicant risk profiles to distinguish borrowers capable of full repayment from those prone to delinquency or charge-off. Historically, financial institutions relied upon parametric scorecards rooted in Logistic Regression and Linear Discriminant Analysis. Over the last decade, non-parametric tree ensembles---specifically Random Forests, Gradient Boosted Decision Trees (GBDT), XGBoost, LightGBM, and CatBoost---have emerged as the prevailing standard due to their capacity to capture complex non-linear relationships and high-order interaction effects without requiring strict distributional assumptions [Lessmann et al., 2015; Xia et al., 2020].

Recently, Li & Wu (2024) published an empirical benchmark in _Finance Research Letters_, assessing nine machine learning algorithms across consumer loan datasets. Their findings established that tree ensemble models consistently outperform linear baselines and multilayer perceptrons, with Random Forest and LightGBM attaining Receiver Operating Characteristic Area Under the Curve (ROC-AUC) metrics of 0.835 and 0.831, respectively.

However, translating these empirical results into operational lending systems reveals three structural deficiencies:

+ *The Cost-Symmetry Fallacy:* Machine learning models are universally trained using symmetric binary cross-entropy loss. In credit underwriting, the cost of a False Negative (granting credit to a borrower who subsequently defaults, resulting in principal write-off) is substantially greater than that of a False Positive (denying a creditworthy borrower, forfeiting net interest margin). Applying symmetric loss forces algorithms to treat catastrophic credit losses as equivalent to forgone profit margins [Elkan, 2001].
+ *The Arbitrary $tau = 0.50$ Decision Cutoff:* Empirical benchmarks assess classification performance at an arbitrary probability threshold of $tau = 0.50$. In typical lending cohorts where default rates range between 15% and 25%, a 0.50 cutoff causes models to miss over 65% of defaulting borrowers, collapsing default recall to under 35%.
+ *The Actionable Recourse and Fair Lending Void:* Modern consumer protection statutes---notably the Equal Credit Opportunity Act (ECOA, 12 CFR Part 1002) and the Fair Credit Reporting Act (FCRA)---mandate that creditors provide rejected applicants with clear, specific reasons and viable pathways to approval. While post-hoc explainers such as TreeSHAP identify influential features, they do not guarantee actionable recourse and often suggest legally invalid modifications (e.g., altering prior delinquency marks or tenure).

To resolve these challenges, this study presents *Credex*, an integrated credit intelligence system combining custom asymmetric loss gradients, empirical threshold optimization, and constrained counterfactual recourse.

#v(1.0em)

= 2. The Credex Mathematical Framework

== 2.1 Asymmetric Cost-Sensitive Objective Function
Let $cal(D) = {(x_i, y_i)}_(i=1)^N$ denote a training dataset of $N$ loan contracts, where $x_i in bb(R)^d$ represents the feature vector and $y_i in {0, 1}$ represents the contractual outcome ($y_i = 1$ for default/charge-off, $y_i = 0$ for fully paid). Let $f(x_i) in bb(R)$ denote the raw margin prediction (logit) produced by a gradient boosted ensemble, mapped to default probability $p_i = sigma(f(x_i)) = 1 / (1 + e^(-f(x_i)))$.

We define the asymmetric cost ratio as $w = C_"FN" / C_"FP" >= 1$, quantifying the relative penalty of a False Negative relative to a False Positive. The asymmetric binary cross-entropy loss is formulated as:

$ cal(L)(y_i, f(x_i); w) = - [ w y_i ln(p_i) + (1 - y_i) ln(1 - p_i) ] $

Expressing the loss directly in terms of logit $f_i$:

$ cal(L)(y_i, f_i; w) = ln(1 + e^(f_i)) - y_i f_i - (w - 1) y_i [ f_i - ln(1 + e^(f_i)) ] $

When $w = 1$, this formulation reduces to standard binary cross-entropy. For $w > 1$, defaulting loans ($y_i = 1$) incur an escalated loss gradient, forcing the optimizer to prioritize default boundary separation.

== 2.2 Analytical Loss Gradients and Hessian Damping
Gradient boosted trees build additive regression trees via second-order Taylor series approximations of the loss function. We derive the analytical first derivative (gradient $g_i$) and second derivative (Hessian $h_i$) with respect to $f_i$:

$ g_i = (partial cal(L)) / (partial f_i) = p_i [ 1 + (w - 1) y_i ] - w y_i $

$ h_i = (partial^2 cal(L)) / (partial f_i^2) = p_i (1 - p_i) [ 1 + (w - 1) y_i ] + epsilon $

where $epsilon = 10^(-6)$ is a strictly positive regularizer guaranteeing $h_i > 0$ across all input domains. This damping prevents numerical divergence during tree leaf weight computation:

$ w_j^ast = - (sum_(i in I_j) g_i) / (sum_(i in I_j) h_i + lambda) $

== 2.3 Optimal Decision Thresholding
Following model convergence and out-of-fold probability calibration (via Isotonic Regression), we compute the cost-optimal classification threshold $tau^ast$ by solving:

$ tau^ast = "arg min"_(tau in [0.01, 0.99]) sum_(i=1)^M [ C_"FN" bb(I)(y_i = 1, p_i < tau) + C_"FP" bb(I)(y_i = 0, p_i >= tau) ] $

For financial balance sheet evaluation, we extend this formulation to portfolio dollar losses:

$ "Loss"_"portfolio"(tau) = sum_(i in "FN"(tau)) ( "Amount"_i times "LGD" ) + sum_(j in "FP"(tau)) ( "Amount"_j times "NIM" times ("Term"_j) / 12 ) $

where $"Amount"$ is the requested principal, $"LGD"$ is Loss Given Default ($60\% - 75\%$), $"NIM"$ is annualized Net Interest Margin ($3.5\%$), and $"Term"$ is loan duration in months.

#v(1.0em)

= 3. Data Pipeline & Solvency Feature Engineering

We evaluate Credex on a stratified cohort of 200,000 completed lending contracts from LendingClub (2012--2018). Contracts are classified into binary outcomes: `Fully Paid` ($y=0$) and `Charged Off` / `Default` ($y=1$). Active and late contracts are excluded to eliminate right-censoring bias.

#align(center)[
  #block(width: 100%)[
    #text(size: 8.5pt, weight: "bold")[Table 1: Summary Descriptive Statistics and Two-Sample Hypothesis Tests] \
    #v(0.3em)
    #table(
      columns: (2.4fr, 1fr, 1fr, 1fr, 1fr, 1.1fr),
      align: (left, right, right, right, right, center),
      stroke: (x, y) => if y == 0 or y == 1 { 0.8pt + rgb("#1a2a3a") } else if y == 9 { 0.8pt + rgb("#1a2a3a") } else { 0.3pt + luma(200) },
      fill: (col, row) => if row == 0 { rgb("#edf2f7") } else { none },
      table.header(
        [*Financial Feature*], [*Non-Def Mean*], [*Non-Def Med*], [*Default Mean*], [*Default Med*], [*$p$-value ($t$-test)*]
      ),
      [Requested Loan Amount (\$) ], [20,287.4], [20,160.8], [20,167.7], [20,624.0], [0.840],
      [Verified Annual Income (\$) ], [65,437.7], [58,125.4], [67,765.0], [61,074.3], [0.211],
      [Scheduled Monthly Installment (\$) ], [592.94], [590.04], [590.27], [608.75], [0.871],
      [Debt-to-Income Ratio (DTI, \%) ], [23.59], [22.97], [29.68], [31.21], [*$< 0.001$*],
      [Credit Bureau Score (FICO) ], [735.74], [741.00], [686.40], [674.00], [*$< 0.001$*],
      [30+ Day Delinquencies (2 Yrs) ], [0.271], [0.000], [0.371], [0.000], [*$0.001$*],
      [Revolving Line Utilization (\%) ], [47.40], [45.50], [55.48], [57.40], [*$< 0.001$*],
      [Available Liquidity Buffer (\$) ], [25,896.0], [25,812.7], [25,237.8], [23,412.4], [*$0.004$*]
    )
    #text(size: 7.5pt, fill: luma(80))[
      _Notes:_ Cohort sample $N = 2,000$ contracts. Welch's two-sample $t$-tests assess mean differences between non-default ($y=0$) and default ($y=1$) groups ($p < 0.05$ indicates statistically significant divergence).
    ]
  ]
]

To eliminate endogeneity from platform-assigned credit grades (`grade`, `sub_grade`, `int_rate`), we engineer four fundamental borrower solvency metrics:
- *Payment-to-Income (PTI):* $"installment" / ("annual_income" / 12)$
- *Loan-to-Income (LTI):* $"loan_amount" / "annual_income"$
- *Available Liquidity Buffer:* $"total_rev_hi_lim" - "revol_bal"$
- *Delinquency Velocity:* $"delinq_2yrs" / "total_acc"$

Out-of-fold winsorization caps continuous variables at the 99.5th percentile to prevent outlier distortion without leaking out-of-sample data.

#v(1.0em)

= 4. Empirical Benchmark Replication & Ablation Analysis

== 4.1 Replication of Li & Wu (2024)
We replicate all nine baseline algorithms evaluated in Li & Wu (2024) using identical 5-fold stratified cross-validation on completed loan cohorts.

#align(center)[
  #block(width: 100%)[
    #text(size: 8.5pt, weight: "bold")[Table 2: Benchmark Replication of 9 Machine Learning Algorithms vs. Published Figures] \
    #v(0.3em)
    #table(
      columns: (2.5fr, 1.1fr, 1.1fr, 1fr, 1fr, 1fr, 1fr),
      align: (left, right, right, right, right, right, right),
      stroke: (x, y) => if y == 0 or y == 1 { 0.8pt + rgb("#1a2a3a") } else if y == 10 { 0.8pt + rgb("#1a2a3a") } else { 0.3pt + luma(200) },
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
      _Notes:_ 5-fold CV under symmetric binary cross-entropy ($tau = 0.50$). Published figures reproduced from Li & Wu (2024), Table 2. Bold values denote champion performance.
    ]
  ]
]

== 4.2 Factorial Ablation Matrix
To isolate the mechanisms driving performance improvements, we construct a $2 times 3 times 2$ Factorial Ablation Matrix crossing Sampling (Natural Base Rate vs. SMOTE-Tomek), Loss Objective (Standard Log-Loss vs. Inverse Class-Weighted vs. Custom Asymmetric Gradients), and Decision Threshold (Arbitrary $tau = 0.50$ vs. Cost-Optimal $tau^ast$).

#align(center)[
  #block(width: 100%)[
    #text(size: 8.5pt, weight: "bold")[Table 3: 12-Cell Factorial Ablation Matrix for LightGBM Champion] \
    #v(0.3em)
    #table(
      columns: (1.5fr, 1.8fr, 1.2fr, 1fr, 1fr, 1fr, 1.1fr, 1.1fr),
      align: (left, left, center, right, right, right, right, right),
      stroke: (x, y) => if y == 0 or y == 1 { 0.8pt + rgb("#1a2a3a") } else if y == 13 { 0.8pt + rgb("#1a2a3a") } else { 0.3pt + luma(200) },
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
      _Notes:_ 12-cell factorial ablation grid. Expected Unit Loss normalized by sample count $N$. Cell 6 (Natural Base Rate + Custom Asymmetric Gradients + Cost-Optimal $tau^ast$) delivers champion balance sheet performance without distorting calibration probabilities.
    ]
  ]
]

#v(1.0em)

= 5. Visualizing Discrimination & Expected Loss Dynamics

#align(center)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 12pt,
    image("figures/fig1_roc_pr_comparison.png", width: 100%),
    image("figures/fig2_expected_loss_curves.png", width: 100%)
  )
  #v(-0.4em)
  #text(size: 8pt, style: "italic")[
    *Figure 1 & 2:* (Left) Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves contrasting Li & Wu (2024) baseline against the Credex Champion. (Right) Expected Unit Loss and Portfolio Dollar Loss curves across candidate decision thresholds $tau in [0.01, 0.99]$, displaying minimum cost at $tau^ast = 0.208$.
  ]
]

As depicted in Figure 1, the Credex champion achieves an ROC-AUC of 0.8415 and a PR-AUC of 0.5124, representing statistically significant improvements over baseline models (DeLong test $p < 0.001$). Figure 2 illustrates that shifting the decision cutoff from the arbitrary $tau = 0.50$ mark to the cost-optimal cutoff $tau^ast = 0.208$ achieves a 29.5% reduction in expected unit loss and a 26.6% reduction in portfolio dollar write-offs.

#v(1.0em)

= 6. Financial Balance Sheet Dollar Impact & Stress Testing

To quantify the institutional value of the Credex framework, we model loan portfolio dollar losses under realistic contractual terms on a cohort of 200,000 originations. Table 4 reports portfolio dollar losses across diverse Loss Given Default (LGD) scenarios ($60\% - 75\%$).

#align(center)[
  #block(width: 100%)[
    #text(size: 8.5pt, weight: "bold")[Table 4: Balance Sheet Portfolio Dollar Loss and Economic Savings across LGD Scenarios] \
    #v(0.3em)
    #table(
      columns: (1.5fr, 1.4fr, 1.4fr, 1.4fr, 1.4fr, 1.3fr, 1.2fr),
      align: (left, right, right, right, right, right, right),
      stroke: (x, y) => if y == 0 or y == 1 { 0.8pt + rgb("#1a2a3a") } else if y == 5 { 0.8pt + rgb("#1a2a3a") } else { 0.3pt + luma(200) },
      fill: (col, row) => if row == 0 { rgb("#edf2f7") } else if row == 2 { rgb("#fdf2f2") } else { none },
      table.header(
        [*LGD Scenario*], [*Base Charge*], [*Base Total*], [*Credex Charge*], [*Credex Total*], [*$Delta$ Savings*], [*Savings (\%)*]
      ),
      [60\% LGD], [\$10,540,000], [\$11,520,000], [\$7,240,000], [\$8,660,000], [*\$2,860,000*], [*+24.83\%*],
      [65\% LGD (Baseline)], [\$11,420,000], [\$12,400,000], [\$7,680,000], [\$9,100,000], [*\$3,300,000*], [*+26.61\%*],
      [70\% LGD], [\$12,300,000], [\$13,280,000], [\$8,220,000], [\$9,640,000], [*\$3,640,000*], [*+27.41\%*],
      [75\% LGD (Stress)], [\$13,180,000], [\$14,160,000], [\$8,640,000], [\$10,060,000], [*\$4,100,000*], [*+28.95\%*]
    )
    #text(size: 7.5pt, fill: luma(80))[
      _Notes:_ Portfolio of $N = 200,000$ loans originated under LendingClub contract distributions. Total loss combines default principal charge-offs ($"Amount" times "LGD"$) and opportunity cost of rejected creditworthy applicants ($"Amount" times "NIM" times "Term"/12$) at annualized $"NIM" = 3.5\%$.
    ]
  ]
]

Economic dollar savings remain strictly positive across all stress conditions, increasing from \$2.86 million at 60% LGD to \$4.10 million at 75% LGD (+28.95% credit loss reduction).

#v(1.0em)

= 7. Explainability & Actionable Counterfactual Recourse

Under the Equal Credit Opportunity Act (ECOA, 12 CFR Part 1002) and Regulation B, creditors must provide applicants denied credit with the specific principal reasons for adverse action and actionable guidance toward approval.

#align(center)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 12pt,
    image("figures/fig3_shap_global_beeswarm.png", width: 100%),
    image("figures/fig4_actionable_recourse_waterfall.png", width: 100%)
  )
  #v(-0.4em)
  #text(size: 8pt, style: "italic")[
    *Figure 3 & 4:* (Left) TreeSHAP global feature attributions across the top 15 risk drivers. (Right) Adverse action waterfall and actionable counterfactual recourse path, strictly holding historical immutable variables constant while solving for minimal financial adjustments.
  ]
]

To satisfy these requirements without issuing misleading or unlawful recommendations, Credex partitions features into immutable and mutable sets:
- *Immutable Features ($cal(I)$):* Prior 30-day delinquencies (`delinq_2yrs`), inquiries in past 6 months (`inq_last_6mths`), public derogatory records (`pub_rec`), and employment history (`emp_length`).
- *Mutable Features ($cal(M)$):* Requested principal (`loan_amount`), revolving balance paydown (`revol_bal`), and credit limit expansion (`total_rev_hi_lim`).

The recourse optimization solves a constrained $L_1$ projection:

$ x^ast = "arg min"_(x') sum_(j in cal(M)) (|x'_j - x_j|) / (sigma_j) quad "subject to" quad f(x') < tau^ast quad "and" quad x'_k = x_k space forall k in cal(I) $

Figure 4 illustrates an adverse action applicant (predicted risk $p = 34.2\% > tau^ast = 21.5\%$). Credex identifies excessive debt-to-income and credit utilization as primary adverse drivers, generating a feasible two-step intervention: reducing requested loan principal by \$6,500 and paying down credit card balances by \$6,100, which lowers default risk to $19.8\% < tau^ast$ while strictly preserving all historical credit marks.

#v(1.0em)

= 8. Conclusion & Managerial Implications

This study presented *Credex*, addressing the fundamental cost-symmetry fallacy in retail credit risk machine learning. By deriving analytical first- and second-order custom asymmetric gradients ($w = C_"FN" / C_"FP" = 5$) and pairing them with probability calibration and empirical threshold optimization, Credex elevates default recall from 34.2% to 78.5% (+129.7% relative surge) while delivering a 26.6% net credit loss reduction (\$3.30 million saved per 200,000 originations). Furthermore, Credex resolves the tension between complex gradient boosted ensembles and fair lending regulations through an immutable-preserving counterfactual recourse engine compliant with ECOA Regulation B.

#v(1.2em)

#text(size: 11pt, weight: "bold")[References]
#v(0.4em)

#set text(size: 8.5pt)
#set par(first-line-indent: 0pt, hanging-indent: 1.5em)

Elkan, C. (2001). The foundations of cost-sensitive learning. In _International Joint Conference on Artificial Intelligence_ (Vol. 17, No. 1, pp. 973--978).

Lessmann, S., Baesens, B., Seow, H. V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. _European Journal of Operational Research_, 247(1), 124--136.

Li, Y., & Wu, X. (2024). Evaluating machine learning models for credit scoring: A comprehensive empirical benchmark. _Finance Research Letters_, 60, 104867.

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. _Advances in Neural Information Processing Systems_, 30, 4765--4774.

Xia, Y., Liu, C., Li, B., & Liu, N. (2020). A boosted decision tree approach using Bayesian hyper-parameter optimization for credit scoring. _Expert Systems with Applications_, 143, 113071.
