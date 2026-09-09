"""
Project 2: Diabetes Disease Progression Prediction (Regression)
-----------------------------------------------------------------
Goal: Predict a quantitative measure of diabetes progression one year
after baseline, using 10 baseline features (age, sex, BMI, blood
pressure, 6 blood serum measurements) — the classic Diabetes dataset
originally used by Efron et al. (2004) "Least Angle Regression".

Why this project:
- Real medical dataset, regression task (predicts a continuous
  disease-severity score, not a category)
- Perfect for demonstrating & TESTING Linear Regression assumptions
- Healthcare-relevant for Optum
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

# Output directory for saved files
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------
# 1. Load data
# ----------------------------
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="disease_progression")

print("Dataset shape:", X.shape)
print(X.describe().T[["mean", "std", "min", "max"]])

# ----------------------------
# 2. Train/test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 3. Train Linear Regression
# ----------------------------
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"\n===== Linear Regression Results =====")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2  : {r2:.4f}")

print("\nCoefficients:")
for feat, coef in zip(X.columns, lin_reg.coef_):
    print(f"  {feat:10s}: {coef:8.2f}")
print(f"  Intercept : {lin_reg.intercept_:.2f}")

# Compare with Random Forest as a benchmark
rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print(f"\n===== Random Forest (benchmark) =====")
print(f"MAE : {mean_absolute_error(y_test, y_pred_rf):.2f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred_rf) ** 0.5:.2f}")
print(f"R2  : {r2_score(y_test, y_pred_rf):.4f}")

# -----------------------------------------------------
# 4. CHECK LINEAR REGRESSION ASSUMPTIONS
#    (implemented manually with numpy/scipy/sklearn - no statsmodels needed)
# -----------------------------------------------------
fitted = lin_reg.predict(X_train)
residuals = (y_train.values - fitted)

print("\n===== Assumption 1: Linearity =====")
print("(visual check saved to linearity_check.png - residuals vs fitted should show no pattern)")

print("\n===== Assumption 2: Independence of errors (Durbin-Watson) =====")
diff_resid = np.diff(residuals)
dw = np.sum(diff_resid ** 2) / np.sum(residuals ** 2)
print(f"Durbin-Watson statistic: {dw:.3f} (~2.0 = no autocorrelation; <1 or >3 is concerning)")

print("\n===== Assumption 3: Homoscedasticity (Breusch-Pagan test) =====")
# Regress squared residuals on the X features; LM = n * R^2 ~ chi2(k)
sq_resid = residuals ** 2
bp_reg = LinearRegression().fit(X_train, sq_resid)
bp_r2 = bp_reg.score(X_train, sq_resid)
n = len(residuals)
k = X_train.shape[1]
lm_stat = n * bp_r2
bp_p = 1 - stats.chi2.cdf(lm_stat, df=k)
print(f"  LM Statistic: {lm_stat:.4f}")
print(f"  p-value     : {bp_p:.4f}")
print("  (p-value > 0.05 => fail to reject H0 => homoscedasticity holds)")

print("\n===== Assumption 4: Normality of residuals (Shapiro-Wilk) =====")
shapiro_stat, shapiro_p = stats.shapiro(residuals)
print(f"  Shapiro-Wilk stat: {shapiro_stat:.4f}, p-value: {shapiro_p:.4f}")
print("  (p-value > 0.05 => residuals approx. normal)")

print("\n===== Assumption 5: No multicollinearity (VIF) =====")
# VIF_i = 1 / (1 - R^2) where R^2 comes from regressing feature i on all other features
vif_scores = []
for i, col in enumerate(X_train.columns):
    other_cols = [c for c in X_train.columns if c != col]
    r2_i = LinearRegression().fit(X_train[other_cols], X_train[col]).score(
        X_train[other_cols], X_train[col]
    )
    vif = 1 / (1 - r2_i) if r2_i < 1 else float("inf")
    vif_scores.append(vif)

vif_data = pd.DataFrame({"feature": X_train.columns, "VIF": vif_scores})
print(vif_data)
print("  (VIF > 5-10 suggests problematic multicollinearity)")

# ----------------------------
# Plots for assumption checks
# ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(fitted, residuals, alpha=0.6)
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("Fitted values")
axes[0].set_ylabel("Residuals")
axes[0].set_title("Residuals vs Fitted (Linearity & Homoscedasticity)")

stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot (Normality of residuals)")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "linearity_check.png"), dpi=120)
print("\nSaved linearity_check.png")

# Save summary
summary = {
    "MAE": mae, "RMSE": rmse, "R2": r2,
    "Durbin_Watson": dw,
    "Breusch_Pagan_p": bp_p,
    "Shapiro_p": shapiro_p,
}
pd.Series(summary).to_csv(os.path.join(OUT_DIR, "results_summary.csv"))
vif_data.to_csv(os.path.join(OUT_DIR, "vif_scores.csv"), index=False)
print("Saved results_summary.csv and vif_scores.csv")
