# Explainable Bank Account Fraud Detection Using Machine Learning

This repository contains an end-to-end AI and machine-learning capstone project focused on bank account fraud detection. It uses the Bank Account Fraud Dataset Suite to build, evaluate, explain, and audit supervised fraud-risk models, with emphasis on fraud-review prioritization, responsible AI, fairness, and business value.

![Project Status](https://img.shields.io/badge/status-capstone%20complete-brightgreen)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Domain](https://img.shields.io/badge/domain-financial%20fraud-red)
![Responsible AI](https://img.shields.io/badge/focus-explainable%20%26%20responsible%20AI-purple)

An end-to-end machine-learning capstone project for prioritizing potentially fraudulent bank-account applications. The project combines temporal model validation, resource-aware hyperparameter tuning, explainability, calibration analysis, fairness auditing, and business-oriented evaluation.

**Author:** Eduardo S. Cudia  
**Program:** Postgraduate Diploma in Artificial Intelligence and Machine Learning  
**Institution:** Asian Institute of Management — School of Executive Education and Lifelong Learning  
**Advisor:** Professor Emeritus Carmen Taglienti  
**Submission:** July 2026

---

## Project Overview

Fraudulent bank-account applications create financial, operational, regulatory, and reputational risks. Traditional rule-based controls and manual reviews remain important, but they may not detect complex or changing fraud patterns efficiently across large application volumes.

This project develops a supervised binary-classification workflow that assigns a fraud-risk score to each application. The model is designed to help fraud analysts prioritize a limited review capacity rather than automatically approve or reject applicants.

### Project objectives

- Compare multiple supervised machine-learning models for fraud detection.
- Optimize model selection using **PR-AUC / Average Precision**, which is appropriate for highly imbalanced data.
- Evaluate operational performance at a **top-10% manual-review capacity**.
- Test temporal generalization using later months and robustness using a separate dataset variant.
- Explain model behavior using coefficients, SHAP, PDP, ICE, and local case analysis.
- Audit outcomes across available age and socioeconomic groups.
- Evaluate business value using transparent, illustrative cost assumptions.
- Save models, configurations, metrics, and reports for reproducibility.

---

## Dataset

The project uses the **Bank Account Fraud Dataset Suite — NeurIPS 2022**.

- **Primary development dataset:** `Base.csv`
- **Robustness stress-test dataset:** `Variant II.csv`
- **Records per dataset:** 1,000,000
- **Original columns:** 32
- **Target:** `fraud_bool`
- **Fraud prevalence:** approximately 1.103%
- **Time field:** `month`, covering months 0–7

The two datasets are analyzed separately and are never merged.

### Data sources

- Kaggle dataset:  
  https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022
- NeurIPS paper:  
  https://proceedings.neurips.cc/paper_files/paper/2022/hash/d9696563856bd350e4e7ac5e5812f23c-Abstract-Datasets_and_Benchmarks.html
- Original repository:  
  https://github.com/feedzai/bank-account-fraud

> The raw CSV files are not required to be committed to this repository. The notebook downloads them through `kagglehub`.

---

## Machine-Learning Design

### Task

Supervised binary classification:

- `0` — legitimate application
- `1` — fraudulent application

### Temporal evaluation

| Data period | Purpose |
|---|---|
| Months 0–5 | Model training |
| Month 6 | Validation, model selection, and threshold selection |
| Month 7 | Final Base test |
| Variant II month 7 | External robustness stress test |

This design reduces the risk of evaluating the model only on randomly mixed historical records and better represents the intended use case of learning from earlier applications and scoring later applications.

### Models evaluated

- Dummy Classifier
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

### Primary evaluation metric

**PR-AUC / Average Precision** was used as the main tuning and model-comparison metric because fraud represents only a small proportion of the observations. Supporting measures include:

- ROC-AUC
- Precision
- Recall
- F1-score
- Confusion matrix
- False-positive and false-negative rates
- Recall and precision at the top review percentage
- Estimated net benefit

---

## Processing and Feature Engineering

The notebook implements a reproducible scikit-learn pipeline with:

- Median imputation for numeric variables
- Most-frequent imputation for categorical variables
- Robust scaling
- One-hot encoding with unknown-category handling
- Feature engineering based on address history, identity consistency, application velocity, balance behavior, and device activity
- Mutual-information analysis
- Random-Forest feature importance
- PCA as an analytical dimensionality-reduction exercise
- Fixed random seed for repeatability

All model preprocessing is fitted within pipelines to reduce training-to-validation leakage.

---

## Key Results

### Model selection

XGBoost produced the strongest cross-validated PR-AUC, but Logistic Regression generalized better to the independent validation month and was selected as the provisional champion.

| Evaluation stage | Result |
|---|---:|
| Best cross-validated model | XGBoost |
| Best cross-validated PR-AUC | 0.140452 |
| Validation champion | Logistic Regression |
| Validation PR-AUC | 0.164799 |
| Validation recall at top 10% | 62.90% |

### Final Base test

| Metric | Result |
|---|---:|
| PR-AUC | 0.198539 |
| ROC-AUC | 0.890668 |
| Recall at top 10% | 67.02% |
| Precision at top 10% | 9.88% |
| Illustrative estimated net benefit | 6,419,500 |

The selected model captured approximately two-thirds of all fraud cases within the highest-risk 10% of applications.

### Variant II robustness test

| Metric | Result |
|---|---:|
| PR-AUC | 0.201282 |
| Recall at top 10% | 66.46% |
| PR-AUC change from Base | +0.002743 |

The near-identical ranking performance indicates good robustness across the two controlled dataset variants. However, calibration and fixed-threshold behavior were less stable, so the model is more suitable for **ranking applications** than for treating raw scores as literal probabilities.

---

## Explainability

The Logistic Regression champion was analyzed through:

- Global coefficient magnitude and direction
- SHAP global feature importance
- SHAP beeswarm plots
- Local waterfall explanations
- Partial Dependence Plots
- Individual Conditional Expectation curves
- Representative true-positive, false-positive, false-negative, and true-negative cases

Important predictive patterns included device reuse, unavailable address history, transaction velocity, phone validity, card ownership, credit indicators, intended balance, housing status, income, and age.

These findings describe model behavior and statistical associations. They should not be interpreted as proof that a characteristic causes fraud.

---

## Responsible AI and Fairness

The project evaluates outcomes across available groups such as:

- Age
- Income bands
- Employment status
- Housing status

Metrics include:

- Demographic parity difference
- Disparate impact ratio
- Equal opportunity difference
- Equalized odds difference
- Selection rate
- True-positive rate
- False-positive rate
- False-negative rate
- Group PR-AUC
- Brier score

Material differences were observed across several age and socioeconomic groups. Race, gender, ethnicity, disability, and other protected characteristics were unavailable and were not inferred.

A reweighting mitigation experiment improved selected parity measures but reduced PR-AUC and recall and worsened other fairness measures. It was therefore not adopted. This demonstrates that mitigation strategies must be evaluated across multiple performance and fairness dimensions.

---

## Important Limitations

1. **Synthetic data:** The BAF suite supports controlled experimentation but does not fully represent a specific institution's customers, controls, products, and fraud typologies.
2. **Class imbalance:** Fraud prevalence is low and changes across months, affecting precision, calibration, and workload.
3. **Probability calibration:** Balanced class weights improved ranking but produced probabilities that were substantially higher than observed fraud prevalence.
4. **Threshold portability:** A threshold selected on Base did not produce the same workload or false-positive rate on Variant II.
5. **Fairness scope:** Several protected attributes were unavailable, so the audit is limited to age and socioeconomic indicators.
6. **Proxy risk:** Age, income, employment, and housing may influence review burdens and require governance oversight.
7. **Potential leakage:** `credit_risk_band` was initially created from dataset-wide quantiles. Removing it had negligible impact, supporting its removal or reconstruction using training-only boundaries.
8. **Illustrative business assumptions:** The net-benefit calculation is a demonstration and must be replaced with validated institutional values.

---

## Recommendation

The model should be used only as a **human-supervised fraud-review prioritization tool**. It should not automatically reject applications.

Before operational deployment:

- Validate the workflow using institution-specific historical data.
- Remove or training-fit `credit_risk_band`.
- Calibrate probabilities using later-period validation data.
- Reassess the threshold using actual fraud prevalence and analyst capacity.
- Monitor technical performance, calibration, fairness, and drift by month and subgroup.
- Provide analysts with understandable reason codes.
- Log overrides and support reconsideration of false positives.
- Compare the interpretable Logistic Regression champion with a calibrated XGBoost alternative.

---

## Repository Structure

A recommended GitHub structure is shown below.

```text
bank-account-fraud-detection/
├── README.md
├── requirements.txt
├── notebooks/
│   └── bank_account_fraud_detection.ipynb
├── data/
│   ├── raw/
│   └── README.md
├── models/
│   ├── champion_logistic_regression.joblib
│   └── candidate_*.joblib
├── configs/
│   └── step4_final_model_config.json
├── reports/
│   ├── step4_tuning_summary.csv
│   ├── step4_validation_model_comparison.csv
│   ├── step4_base_test_results.csv
│   ├── step4_variant_ii_results.csv
│   ├── step4_robustness_gap.csv
│   ├── step4_pr_auc_confidence_intervals.csv
│   └── step5/
├── presentations/
│   ├── technical_presentation.pptx
│   └── business_presentation.pptx
└── docs/
    ├── capstone_report.pdf
    └── challenges_adjustments_recommendations.pdf
```

The notebook automatically creates the `models/`, `configs/`, and `reports/` directories when the relevant sections are executed.

---

## Installation

### Option 1: Google Colab

1. Upload or open `bank_account_fraud_detection.ipynb` in Google Colab.
2. Use a runtime with sufficient memory for the two one-million-record datasets.
3. Run the notebook from top to bottom.
4. Allow the notebook to install or import the required Python packages.
5. The dataset is retrieved through `kagglehub`.

### Option 2: Local environment

```bash
git clone https://github.com/YOUR-USERNAME/bank-account-fraud-detection.git
cd bank-account-fraud-detection

python -m venv .venv
```

Activate the virtual environment.

**Windows**

```bash
.venv\Scripts\activate
```

**macOS or Linux**

```bash
source .venv/bin/activate
```

Install the packages:

```bash
pip install numpy pandas scipy matplotlib scikit-learn xgboost shap joblib kagglehub jupyter
```

Start Jupyter:

```bash
jupyter lab
```

Open:

```text
notebooks/bank_account_fraud_detection.ipynb
```

---

## Reproducibility

The project uses:

- Random seed: `42`
- Training months: `0–5`
- Validation month: `6`
- Test month: `7`
- Primary review policy: `10%`
- Resource-aware hyperparameter tuning
- Saved candidate models and champion pipeline
- Saved configuration in JSON
- Saved technical, business, robustness, calibration, and fairness reports
- Reload verification of the saved champion model

The full search spaces and final parameters are documented in the notebook and saved configuration file.

---

## Computational Considerations

The project was developed in Google Colab with resource-aware adjustments:

- A 150,000-record development sample was used for hyperparameter tuning.
- Three-fold stratified cross-validation was used for tuning.
- Large search spaces used randomized search.
- SHAP used representative background and evaluation samples.
- Logistic Regression convergence settings were revised.
- The final champion was refitted on the complete Base training period.

These adjustments reduced timeout and memory risks while retaining independent temporal validation and final testing.

---

## Business Interpretation

The model is intended to improve manual-review efficiency by ranking the most suspicious applications first. At the top-10% review capacity, it captured 67.02% of fraud on the Base test set.

The estimated net benefit of 6.42 million is based on the notebook's illustrative assumptions:

- Estimated loss per fraud: 10,000
- Manual review cost: 100
- False-positive cost: 250

These values must be replaced with institutionally validated assumptions before a real business case is approved.

---

## Deliverables

- Final executable Jupyter notebook
- Saved candidate and champion models
- Model configuration file
- Technical evaluation reports
- Robustness and confidence-interval reports
- Explainability and fairness reports
- Technical presentation
- Business presentation
- Capstone compliance review
- Challenges, adjustments, and recommendations document

---

## References

Jesus, S. (2022). *Bank Account Fraud Dataset Suite (NeurIPS 2022)*. Kaggle.  
https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022

Jesus, S., Pombal, J., Alves, D., Cruz, A., Saleiro, P., Ribeiro, R. P., Gama, J., & Bizarro, P. (2022). *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation*. Advances in Neural Information Processing Systems, 35.  
https://proceedings.neurips.cc/paper_files/paper/2022/hash/d9696563856bd350e4e7ac5e5812f23c-Abstract-Datasets_and_Benchmarks.html

Feedzai. (2022). *Bank Account Fraud Dataset Suite*. GitHub.  
https://github.com/feedzai/bank-account-fraud

Scikit-learn documentation:  
https://scikit-learn.org/stable/

XGBoost documentation:  
https://xgboost.readthedocs.io/

SHAP documentation:  
https://shap.readthedocs.io/

Fairlearn documentation:  
https://fairlearn.org/

---

## Ethical Use Notice

This repository is an academic capstone project. The model must not be used to make fully automated decisions about real applicants. Any institutional implementation requires legal, compliance, privacy, model-risk, cybersecurity, and fairness review, together with documented human oversight.

---

## Acknowledgement

This project was completed as part of the Postgraduate Diploma in Artificial Intelligence and Machine Learning at the Asian Institute of Management.
