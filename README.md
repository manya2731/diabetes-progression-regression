# Diabetes Disease Progression Prediction (Regression)

Predicts a quantitative measure of diabetes progression one year after baseline, using 10 baseline variables (age, sex, BMI, average blood pressure, and 6 blood serum measurements) for 442 diabetes patients (classic dataset from Efron et al., 2004, built into scikit-learn).

## Why this matters

A continuous progression score (rather than a yes/no label) lets clinicians prioritize patients by expected severity, not just flag them. This project also rigorously checks the 5 assumptions of Linear Regression rather than just reporting an R² — a step many beginner projects skip.

## Approach

1. Load data (already mean-centered / scaled by the dataset provider)
2. 80/20 train-test split
3. Fit Linear Regression; benchmark against Random Forest
4. Test all 5 Linear Regression assumptions on the residuals
5. Visualize residuals-vs-fitted and a Q-Q plot

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 42.79 | 53.85 | 0.453 |
| Random Forest | 44.58 | 54.76 | 0.434 |

## Assumption checks (rigorous validation, not just R²)

| Assumption | Test used | Result | Verdict |
|---|---|---|---|
| Linearity | Residuals vs Fitted plot | No strong curve pattern | Roughly holds |
| Independence of errors | Durbin-Watson | 1.79 (close to 2.0) | Holds |
| Homoscedasticity | Breusch-Pagan test | p = 0.035 (< 0.05) | Violated (mild heteroscedasticity) |
| Normality of residuals | Shapiro-Wilk test | p = 0.64 (> 0.05) | Holds |
| No multicollinearity | Variance Inflation Factor (VIF) | s1 VIF=55, s2 VIF=36, s3 VIF=14 | Violated |

**Takeaway:** the blood-serum features (s1–s5) are highly correlated with each other (they're all lipid/cholesterol-related measurements), which inflates their VIF. This doesn't break predictions (R² is still fine) but it means individual coefficients (e.g., "s1 has coefficient -931") can't be trusted for interpretation — a classic real-world caveat.

## How to run

```bash
pip install -r requirements.txt
python diabetes_progression_regressor.py
```

## Tech stack

Python, scikit-learn, pandas, numpy, scipy, matplotlib

## Possible extensions

- Ridge/Lasso regression to handle multicollinearity (regularization)
- Drop or combine correlated s1-s5 features (PCA)
- Try polynomial features for non-linear relationships
