# Customer Churn Prediction (Telecom)

Predicting which telecom customers are likely to cancel, and explaining *why*, so a retention team knows who to contact and what to offer. Built on 7,043 customers from the IBM Telco dataset, where about 26.5% of customers churned.

<img width="790" height="940" alt="shap_summary" src="https://github.com/user-attachments/assets/dfd828eb-25ef-4221-a59a-51e92e2e4e5a" />


## Key findings

- **Contract type is the biggest driver.** Month-to-month customers churn far more than one- or two-year customers; a two-year contract is the single strongest signal that a customer will stay.
- **New customers are the highest risk.** Short tenure pushes churn risk up sharply; risk falls the longer someone stays.
- **Fiber optic and electronic-check customers churn more,** even after accounting for contract and tenure, which points to possible price/value or service-quality issues worth investigating.
- **Add-ons like Online Security and Tech Support are associated with lower churn.**

<img width="989" height="490" alt="contract_vs_churn" src="https://github.com/user-attachments/assets/ad664437-7f22-4fd2-b4a6-012825049750" />

**Business takeaway:** the highest-value retention target is a new, month-to-month customer, especially on fiber. Offering a discounted move to a 1-year contract or bundling Tech Support/Online Security is the lever the model points to.

## Model results (held-out test set, 1,409 customers, 374 churners)

| Model | ROC-AUC | PR-AUC | Churners caught (recall) | Precision |
|---|---|---|---|---|
| Logistic Regression | 0.842 | 0.632 | 57% | 0.66 |
| Random Forest | 0.843 | 0.657 | 52% | 0.68 |
| **XGBoost (tuned)** | **0.846** | **0.663** | **79% (297 / 374)** | 0.52 |

The tuned XGBoost model uses class weighting (`scale_pos_weight`) to prioritize catching churners. It finds **297 of 374** churners vs. 212 for Logistic Regression, at the cost of more false positives. For retention campaigns that trade-off usually makes sense, since contacting a loyal customer is cheap compared to losing one.

## Approach

1. **Data cleaning:** converted `TotalCharges` to numeric (11 blank values for brand-new customers, imputed), dropped the ID column, encoded the target.
2. **EDA:** churn by contract type, tenure, monthly charges, and services.
3. **Encoding and scaling:** one-hot encoding for categorical features, standard scaling for Logistic Regression.
4. **Models:** Logistic Regression, Random Forest, and XGBoost with class weighting.
5. **Hyperparameter tuning:** `GridSearchCV` over 729 XGBoost configurations (3-fold CV, optimizing F1).
6. **Evaluation:** ROC-AUC, PR-AUC, precision/recall/F1 on the churn class, confusion matrices, and threshold analysis.
7. **Explainability:** SHAP summary and dependence plots to show how each feature pushes an individual prediction up or down.

## Dataset

[Telco Customer Churn (IBM sample data, Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): 7,043 customers with demographics, account details (tenure, contract, billing), and subscribed services.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook customer_churn_prediction.ipynb
```

The notebook downloads the data automatically with `kagglehub`.

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · SHAP · Matplotlib · Seaborn

## Next steps

- Choose the decision threshold on a validation split using a simple cost model (retention offer cost vs. customer lifetime value).
- Score customers into risk tiers (high / medium / low) and present them in a Power BI or Tableau dashboard.

---
**Author:** Indraneel Mannava · [LinkedIn](https://www.linkedin.com/in/indraneel-sarma-mannava/)
