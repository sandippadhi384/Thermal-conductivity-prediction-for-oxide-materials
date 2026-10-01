import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import tensorflow
from keras.layers import Dense, Input
from keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

tensorflow.keras.utils.set_random_seed(42)  # reproducible output with random state

# 1. Load cleaned features file obtained from feature extraction code
dataset4 = pd.read_csv("cleaned_features.csv")

x1 = dataset4.drop(["Chemical Formula", "Thermal Conductivity(W/mK)"], axis=1)
measured = dataset4["Chemical Formula"] != "SrSnO3"

x = x1[measured]                                        # features of measured compounds
y = dataset4.loc[measured, "Thermal Conductivity(W/mK)"]  # measured thermal conductivity
z = x1[~measured]                                       # features of SrSnO3 (to predict)
srsno3_temperature = dataset4.loc[~measured, "Temperature(K)"].values

feature_names = x1.columns.tolist()
print("Measured data:", x.shape, " SrSnO3:", z.shape)

# 2. Train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
# 3. Remove outliers from the training set using z score method
z_scores = np.abs(stats.zscore(y_train))
train_mask = z_scores < 3  # keep only samples within 3 standard deviations
x_train = x_train[train_mask]
y_train = y_train[train_mask]
print("Training samples after outlier removal:", x_train.shape[0])

# 4. Imputation and scaling
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test)
z = imputer.transform(z)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
x_srca2 = sc.transform(z)

# 5. Artificial neural network model

n_features = x_train.shape[1]

ann = Sequential()
ann.add(Input(shape=(n_features,)))
ann.add(Dense(n_features, activation="relu"))  # first layer: one unit per feature
ann.add(Dense(256, activation="relu"))
ann.add(Dense(128, activation="relu"))
ann.add(Dense(64, activation="relu"))
ann.add(Dense(32, activation="relu"))
ann.add(Dense(16, activation="relu"))
ann.add(Dense(1, activation="relu"))  # thermal conductivity is non-negative so using relu

ann.compile(optimizer="adam", loss="mse", metrics=["mae"])
ann.summary()

early_stopping = EarlyStopping(  #to avoid overfitting
    monitor="val_loss",          # metric to monitor
    patience=10,                 # epochs to wait for improvement
    restore_best_weights=True,   # restore weights from the best epoch
)
history = ann.fit(x_train, y_train, batch_size=20, epochs=1000,
                  validation_data=(x_test, y_test), callbacks=[early_stopping])

ann.save("ann_model.keras")
pd.DataFrame(history.history).to_csv("ANN_training_history.csv", index_label="epoch")

# ---------------------------------------------------------------------------
# 6. Validation
# ---------------------------------------------------------------------------
y_train_pred = ann.predict(x_train).reshape(-1)
y_test_pred = ann.predict(x_test).reshape(-1)

for name, true, pred in [("Train", y_train, y_train_pred), ("Test", y_test, y_test_pred)]:
    print(f"ANN {name:5s}  R2 = {r2_score(true, pred):.4f}   "
          f"MAE = {mean_absolute_error(true, pred):.4f}   "
          f"RMSE = {np.sqrt(mean_squared_error(true, pred)):.4f}")

np.savetxt("y_train_ANN.csv", y_train, delimiter=",")
np.savetxt("y_train_pred_ANN.csv", y_train_pred, delimiter=",")
np.savetxt("y_test_ANN.csv", y_test, delimiter=",")
np.savetxt("y_test_pred_ANN.csv", y_test_pred, delimiter=",")

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
plt.title("ANN")
plt.legend()
plt.tight_layout()
plt.savefig("ANN_parity_plot.png", dpi=300)

# ---------------------------------------------------------------------------
# 7. Prediction for SrSnO3
# ---------------------------------------------------------------------------
y_pred2 = ann.predict(x_srca2).reshape(-1)
extracted = pd.DataFrame({"Temperature": srsno3_temperature, "thermal_conductivity": y_pred2})
print(extracted)
extracted.to_csv("ANN prediction.csv", index=False)
