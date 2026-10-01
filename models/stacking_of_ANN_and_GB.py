import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow import keras

# 1. Load cleaned features from feature extraction code
dataset4 = pd.read_csv("cleaned_features.csv")

x1 = dataset4.drop(["Chemical Formula", "Thermal Conductivity(W/mK)"], axis=1)
measured = dataset4["Chemical Formula"] != "SrSnO3"

x = x1[measured]                                        # features of measured compounds
y = dataset4.loc[measured, "Thermal Conductivity(W/mK)"]  # measured thermal conductivity
z = x1[~measured]                                       # features of SrSnO3 (to predict)
srsno3_temperature = dataset4.loc[~measured, "Temperature(K)"].values

# ---------------------------------------------------------------------------
# 2. Train/test split (same as the ANN and GB scripts)
# ---------------------------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# ---------------------------------------------------------------------------
# 3. Remove outliers from the training set
# ---------------------------------------------------------------------------
z_scores = np.abs(stats.zscore(y_train))
train_mask = z_scores < 3
x_train = x_train[train_mask]
y_train = y_train[train_mask]

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
# 5. Load the trained base models
# ---------------------------------------------------------------------------
ann = keras.models.load_model("ann_model.keras")
gb = joblib.load("gb_model.joblib")

# ---------------------------------------------------------------------------
# 6. Stacking: SVR meta-model on the base-model predictions
# ---------------------------------------------------------------------------
gb_pred = gb.predict(x_test)
ann_pred = ann.predict(x_test).reshape(-1)

# Combine the predictions of the base models into a single feature matrix
X_val_meta = np.column_stack((gb_pred, ann_pred))

# Train the meta-model on the combined feature matrix and the target values
# (as in the paper, the meta-model is fitted on the test-set predictions)
meta_model = SVR(kernel="rbf")
meta_model.fit(X_val_meta, y_test)

joblib.dump(meta_model, "stacking_meta_svr.joblib")

# ---------------------------------------------------------------------------
# 7. Validation
# ---------------------------------------------------------------------------
y_test_pred = meta_model.predict(X_val_meta)

X_train_meta = np.column_stack((gb.predict(x_train), ann.predict(x_train).reshape(-1)))
y_train_pred = meta_model.predict(X_train_meta)

for name, true, pred in [("Train", y_train, y_train_pred), ("Test", y_test, y_test_pred)]:
    print(f"Stacking {name:5s}  R2 = {r2_score(true, pred):.4f}   "
          f"MAE = {mean_absolute_error(true, pred):.4f}   "
          f"RMSE = {np.sqrt(mean_squared_error(true, pred)):.4f}")

pd.DataFrame({"Experimental thermal conductivity(W/mK)": y_test.values,
              "predicted thermal conductivity(W/mK)": y_test_pred}
             ).to_csv("test prediction data.csv", index=False)
pd.DataFrame({"Experimental thermal conductivity(W/mK)": y_train.values,
              "predicted thermal conductivity(W/mK)": y_train_pred}
             ).to_csv("train prediction data.csv", index=False)

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
plt.title("Stacking (ANN + GB → SVR)")
plt.legend()
plt.tight_layout()
plt.savefig("Stacking_parity_plot.png", dpi=300)

# ---------------------------------------------------------------------------
# 8. Prediction for SrSnO3
# ---------------------------------------------------------------------------
gb_pred_new = gb.predict(x_srca2)
ann_pred_new = ann.predict(x_srca2).reshape(-1)

X_new_meta = np.column_stack((gb_pred_new, ann_pred_new))
y_pred2 = meta_model.predict(X_new_meta)

extracted = pd.DataFrame({"Temperature": srsno3_temperature,
                          "ANN": ann_pred_new,
                          "GB": gb_pred_new,
                          "thermal_conductivity": y_pred2})
print(extracted)
extracted.to_csv("Stacking prediction.csv", index=False)
