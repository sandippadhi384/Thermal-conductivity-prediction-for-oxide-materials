import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Load cleaned features from feature extraction code
dataset4 = pd.read_csv("cleaned_features.csv")

x1 = dataset4.drop(["Chemical Formula", "Thermal Conductivity(W/mK)"], axis=1)
measured = dataset4["Chemical Formula"] != "SrSnO3"

x = x1[measured]                                        # features of measured compounds
y = dataset4.loc[measured, "Thermal Conductivity(W/mK)"]  # measured thermal conductivity
z = x1[~measured]                                       # features of SrSnO3 (to predict)
srsno3_temperature = dataset4.loc[~measured, "Temperature(K)"].values

feature_names = x1.columns.tolist()
print("Measured data:", x.shape, " SrSnO3:", z.shape)

# ---------------------------------------------------------------------------
# 2. Train/test split
# ---------------------------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# ---------------------------------------------------------------------------
# 3. Remove outliers from the training set
# ---------------------------------------------------------------------------
z_scores = np.abs(stats.zscore(y_train))
train_mask = z_scores < 3  # keep only samples within 3 standard deviations
x_train = x_train[train_mask]
y_train = y_train[train_mask]
print("Training samples after outlier removal:", x_train.shape[0])

# ---------------------------------------------------------------------------
# 4. Imputation and scaling
# ---------------------------------------------------------------------------
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test)
z = imputer.transform(z)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
x_srca2 = sc.transform(z)

# ---------------------------------------------------------------------------
# 5. Gradient boosting model
# ---------------------------------------------------------------------------
gb = GradientBoostingRegressor(learning_rate=0.2, max_depth=3, random_state=42)
gb.fit(x_train, y_train)

joblib.dump(gb, "gb_model.joblib")

# ---------------------------------------------------------------------------
# 6. Validation
# ---------------------------------------------------------------------------
y_train_pred = gb.predict(x_train)
y_test_pred = gb.predict(x_test)

for name, true, pred in [("Train", y_train, y_train_pred), ("Test", y_test, y_test_pred)]:
    print(f"GB {name:5s}  R2 = {r2_score(true, pred):.4f}   "
          f"MAE = {mean_absolute_error(true, pred):.4f}   "
          f"RMSE = {np.sqrt(mean_squared_error(true, pred)):.4f}")

np.savetxt("y_train_GB.csv", y_train, delimiter=",")
np.savetxt("y_train_pred_GB.csv", y_train_pred, delimiter=",")
np.savetxt("y_test_GB.csv", y_test, delimiter=",")
np.savetxt("y_test_pred_GB.csv", y_test_pred, delimiter=",")

# Parity plot
plt.figure(figsize=(5, 5))
plt.scatter(y_train, y_train_pred, s=12, alpha=0.6, label=f"Train (R$^2$ = {r2_score(y_train, y_train_pred):.3f})")
plt.scatter(y_test, y_test_pred, s=12, alpha=0.8, label=f"Test (R$^2$ = {r2_score(y_test, y_test_pred):.3f})")
lim = max(y_train.max(), y_test.max(), y_train_pred.max(), y_test_pred.max()) * 1.05
plt.plot([0, lim], [0, lim], "k--", lw=1)
plt.xlim(0, lim)
plt.ylim(0, lim)
plt.xlabel("Experimental thermal conductivity (W/mK)")
plt.ylabel("Predicted thermal conductivity (W/mK)")
plt.title("Gradient boosting")
plt.legend()
plt.tight_layout()
plt.savefig("GB_parity_plot.png", dpi=300)

# ---------------------------------------------------------------------------
# 7. Prediction for SrSnO3
# ---------------------------------------------------------------------------
y_pred3 = gb.predict(x_srca2)
extracted = pd.DataFrame({"Temperature": srsno3_temperature, "thermal_conductivity": y_pred3})
print(extracted)
extracted.to_csv("GB prediction.csv", index=False)
