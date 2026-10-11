# Credit Risk Model with Early Warning Dashboard

**Author:** Archie P. H. Sinaga · [GitHub](https://github.com/archiesinaga)
**Repo:** https://github.com/archiesinaga/credit-risk-model
**Live demo:** _(https://credit-risk-archie.streamlit.app)_

> **Data note:** this project uses the public Kaggle dataset *Give Me Some Credit*. The data is real but anonymized and comes from another country. It is **not** Indonesian bank data. Banks do not publish loan-level data, so the method transfers to a bank's own portfolio, the data does not.

## 1. The problem and the decision

- **Borrower:** an individual consumer borrower with a credit history.
- **Default:** serious delinquency (90+ days past due) within 2 years (`SeriousDlqin2yrs = 1`).
- **Decision supported:** approve, review, or reject, and which existing borrowers to flag early.
- **Output:** a calibrated probability of default (PD), a risk grade A to E, and the top reasons behind each score.

## 2. The data and its limits

- 150,000 borrowers, 10 features, default rate 6.68%.
- Split 70/15/15 (train / validation / test), stratified, `random_state=42`. The test set is used once, at the end.
- Missing values: `MonthlyIncome` 29,731 (about 20%), `NumberOfDependents` 3,924.
- Findings that shaped the cleaning:
  - **Special codes 96 and 98** in the three late-payment columns (269 rows) are not real counts. Default rate on these rows is **54.6%** versus 6.6% elsewhere, so they are kept as a flag (`late_code_flag`), not deleted.
  - **28,877 rows have DebtRatio > 10**, and 26,771 of them have missing income. In this dataset DebtRatio holds an absolute debt amount, not a ratio, when income is missing. These rows get an `income_missing` flag and a median DebtRatio.
  - Extreme values (utilization, DebtRatio, income) are capped at the 99th percentile.
- **All imputation values and caps are computed on the training set only** and applied to validation and test, to avoid data leakage.
- Limitation: single random split, no confidence intervals, and a foreign anonymized population.

## 3. Method in 5 lines

1. Clean and flag data with rules fitted on train only (`src/cleaning.py`).
2. Baseline: logistic regression with `class_weight="balanced"` (transparent, scorecard-style).
3. Challenger: XGBoost with `scale_pos_weight` and early stopping on validation.
4. Calibrate PD with isotonic regression, because class weighting inflates raw probabilities (mean raw PD about 0.32 to 0.34 versus a real default rate of 0.0668).
5. Choose the cutoff by business cost, convert PD to grades A to E, explain with SHAP.

## 4. Results (test set, never used for tuning)

| Model | AUC | Gini | KS |
|---|---|---|---|
| Logistic regression | 0.8540 | 0.7080 | 0.5480 |
| XGBoost | **0.8613** | **0.7226** | **0.5699** |

The simple model is only about 0.007 AUC behind the boosted tree. A bank can trade a little accuracy for transparency.

**Calibration:** mean predicted PD on test is 0.066 against an actual default rate of 0.0668. Calibration curves are in `reports/calibration_test.png`.

**Cutoff by business cost.** Missing a defaulter is assumed to cost 5 times more than rejecting a good customer (5:1). The cutoff was chosen on validation and then frozen:

| Model | Cutoff | Precision | Recall | Flagged | Cost saving vs approving everyone |
|---|---|---|---|---|---|
| Logistic regression | 0.140 | 0.361 | 0.539 | 10.0% | 34.8% |
| XGBoost | 0.158 | 0.354 | 0.570 | 10.8% | **36.2%** |

The optimal cutoff moves with the cost ratio: about 0.25 at 3:1, 0.16 at 5:1 and 0.07 at 10:1 (see `reports/cutoff_cost.png`). The cost curve is flat near its minimum, so the choice is not fragile.

**Risk grades (XGBoost, test set).** Grade limits were set on validation and not changed afterwards.

| Grade | PD range | Suggested action | Borrowers | Share | Mean PD | Actual default rate |
|---|---|---|---|---|---|---|
| A | < 2% | Approve | 11,938 | 53.1% | 0.94% | 1.11% |
| B | 2% to 5% | Approve | 3,123 | 13.9% | 3.11% | 3.27% |
| C | 5% to 10% | Approve with conditions | 3,958 | 17.6% | 7.10% | 6.82% |
| D | 10% to 20% | Manual review | 1,230 | 5.5% | 12.87% | 13.90% |
| E | > 20% | Reject or require collateral | 2,251 | 10.0% | 37.09% | 36.78% |

Actual default rates rise from A to E, so the model ranks borrowers correctly. Grade E is about 10% of borrowers but roughly 55% of all defaults. The model slightly understates risk in grades A and D.

## 5. Key SHAP findings

Global importance (mean absolute SHAP), in order: revolving utilization, 30 to 59 days late, 90+ days late, age, 60 to 89 days late, open credit lines, debt ratio, real estate loans, monthly income, dependents.

- Credit utilization is the strongest driver, about twice the second feature.
- The three late-payment features push risk up sharply as counts grow.
- Younger borrowers get higher risk, older borrowers lower risk.
- Number of dependents has almost no effect.
- Per-borrower explanations are in `reports/shap_borrower_D.png`. A grade D borrower is not certain to default: about 14% of grade D borrowers did. A grade states a probability, not a verdict.

SHAP values are on the raw model's log-odds scale. Use them for the order and direction of effects, not as percentage points of PD.

## 6. Dashboard

Streamlit app with three pages:

1. **Portfolio overview:** borrowers, default rate, share per grade, default rate per grade.
2. **Early warning list:** highest-risk borrowers with grade and top 3 risk reasons from SHAP.
3. **Score a borrower:** form that returns PD, grade, and suggested action.

PD is displayed with a 99% maximum, because isotonic calibration returns 1.0 for the top score bucket.

## 7. How to run it

```bash
git clone https://github.com/archiesinaga/credit-risk-model.git
cd credit-risk-model
python -m venv .venv
source .venv/bin/activate          # Windows cmd: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

1. Download `cs-training.csv` from the Kaggle competition *Give Me Some Credit* and place it in `data/raw/`. **Do not open or re-save it in Excel.** A non-English Excel locale reads the decimal point as a thousands separator and silently corrupts the numbers.
2. Run the notebooks in this order: `02_features` (split and cleaning), `01_eda`, `03_models`, `04_evaluation`, `05_grades_shap`.
3. Start the dashboard:

```bash
python -m streamlit run app/streamlit_app.py
```

Only the dashboard needs `requirements.txt`. Package versions are pinned because saved `joblib` models may not load across different scikit-learn or XGBoost versions.

##