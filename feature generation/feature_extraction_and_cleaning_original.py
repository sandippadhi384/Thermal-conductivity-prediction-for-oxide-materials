"""
Feature extraction and data cleaning (original workflow)
========================================================

Original code from the notebook "Stacking ANN+GB for oxide dataset for feature",
collected into a single script.

Steps
-----
1. Load the dataset and append SrSnO3 (temperatures to be predicted, target = 0)
2. Convert chemical formulas to compositions
3. Generate matminer features: ElementProperty (DEML, Magpie), ElementFraction,
   Stoichiometry, AtomicOrbitals, YangSolidSolution, Meredig
4. Drop columns with more than 250 missing values
5. One-hot encode LUMO/HOMO element and character columns
6. Drop columns with zero standard deviation
7. Drop duplicate columns (identical content)
8. Save the cleaned dataset

Requirements
------------
    pip install pymatgen matminer openpyxl
"""
import numpy as np
import pandas as pd

from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.composition import ElementFraction
from matminer.featurizers.composition import Stoichiometry
from matminer.featurizers.composition import AtomicOrbitals
from matminer.featurizers.composition import YangSolidSolution
from matminer.featurizers.composition import Meredig
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
dataset = pd.read_excel("Final ML Datasets - Copy.xlsx")
dataset2 = dataset.copy()

# Compound to be predicted (SrSnO3) with placeholder thermal conductivity = 0
srca = pd.DataFrame({"temperature1(K)": [80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300,
                                         303, 323, 373, 423, 473, 523, 573, 623, 673, 723, 773, 823, 873]})
srca["Chemical Formula"] = "SrSnO3"
srca["Temperature(K)"] = [80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300,
                          303, 323, 373, 423, 473, 523, 573, 623, 673, 723, 773, 823, 873]
srca["Thermal Conductivity(W/mK)"] = 0
srca.drop(columns=["temperature1(K)"], axis=1, inplace=True)

dataset2 = pd.concat([dataset2, srca])
dataset2.reset_index(drop=True, inplace=True)
print(dataset2)

# ---------------------------------------------------------------------------
# 2. Formula -> composition
# ---------------------------------------------------------------------------
str = StrToComposition()
dataset2 = str.featurize_dataframe(dataset2, "Chemical Formula")

# ---------------------------------------------------------------------------
# 3. Feature generation
# ---------------------------------------------------------------------------
# Element property (DEML preset)
ep_feat = ElementProperty.from_preset(preset_name="deml")
dataset3 = ep_feat.featurize_dataframe(dataset2, col_id="composition", ignore_errors=True)

# Element property (Magpie preset)
ep_feat = ElementProperty.from_preset(preset_name="magpie")
dataset3 = ep_feat.featurize_dataframe(dataset3, col_id="composition")

# 1. Element fractions (composition one-hot encoding)
ef_feat = ElementFraction()
dataset3 = ef_feat.featurize_dataframe(dataset3, "composition", ignore_errors=True)

# 2. Stoichiometry
ef_feat = Stoichiometry()
dataset3 = ef_feat.featurize_dataframe(dataset3, "composition", ignore_errors=True)

# 3. Atomic orbitals (HOMO/LUMO levels)
ao_feat = AtomicOrbitals()
dataset3 = ao_feat.featurize_dataframe(dataset3, "composition", ignore_errors=True)

# 4. Yang solid solution descriptors (mixing entropy, etc.)
yang_feat = YangSolidSolution()
dataset3 = yang_feat.featurize_dataframe(dataset3, "composition", ignore_errors=True)

# 5. Meredig features (classic hand-crafted descriptors)
meredig_feat = Meredig()
dataset3 = meredig_feat.featurize_dataframe(dataset3, "composition", ignore_errors=True)
print(dataset3)

# ---------------------------------------------------------------------------
# 4. Drop columns with more than 250 missing values
# ---------------------------------------------------------------------------
nan_list = []
for i in range(5, len(dataset3.columns)):
    if dataset3[dataset3.columns[i]].isnull().sum() > 250:
        nan_list.append(i)
print("Columns with > 250 missing values:", nan_list)

dataset3.drop(dataset3.columns[nan_list], axis=1, inplace=True)

# ---------------------------------------------------------------------------
# 5. One-hot encoding of categorical columns
# ---------------------------------------------------------------------------
# LUMO_element
enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded = enc.fit_transform(dataset3[["LUMO_element"]])
encoded_df = pd.DataFrame(encoded, columns=enc.get_feature_names_out(["LUMO_element"]))
dataset3 = pd.concat([dataset3.drop(columns=["LUMO_element"]), encoded_df], axis=1)

# LUMO_character
enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded = enc.fit_transform(dataset3[["LUMO_character"]])
encoded_df = pd.DataFrame(encoded, columns=enc.get_feature_names_out(["LUMO_character"]))
dataset3 = pd.concat([dataset3.drop(columns=["LUMO_character"]), encoded_df], axis=1)

# HOMO_element
enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded = enc.fit_transform(dataset3[["HOMO_element"]])
encoded_df = pd.DataFrame(encoded, columns=enc.get_feature_names_out(["HOMO_element"]))
dataset3 = pd.concat([dataset3.drop(columns=["HOMO_element"]), encoded_df], axis=1)

# HOMO_character
enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded = enc.fit_transform(dataset3[["HOMO_character"]])
encoded_df = pd.DataFrame(encoded, columns=enc.get_feature_names_out(["HOMO_character"]))
dataset3 = pd.concat([dataset3.drop(columns=["HOMO_character"]), encoded_df], axis=1)

print(dataset3.isnull().sum())

# ---------------------------------------------------------------------------
# 6. Drop columns with zero standard deviation
# ---------------------------------------------------------------------------
std_zero_columns = []
for i in range(8, len(dataset3.columns)):
    if dataset3.iloc[:, i].std() == 0:
        std_zero_columns.append(dataset3.columns[i])
print("Zero-std columns:", std_zero_columns)

for i in std_zero_columns:
    dataset3.drop(i, axis=1, inplace=True)

dataset4 = dataset3.copy()

# ---------------------------------------------------------------------------
# 7. Drop duplicate columns (identical content)
# ---------------------------------------------------------------------------
dup_cols = []
for i in range(len(dataset3.columns)):
    for j in range(i + 1, len(dataset3.columns)):
        if dataset3.iloc[:, i].equals(dataset3.iloc[:, j]):
            dup_cols.append(dataset3.columns[j])

print("Duplicate columns by content:", dup_cols)
print(np.size(dup_cols))

dataset4.drop(dup_cols, axis=1, inplace=True)
print(dataset4)

# ---------------------------------------------------------------------------
# 8. Save cleaned dataset
# ---------------------------------------------------------------------------
dataset4.drop(columns=["composition"]).to_csv("cleaned_features.csv", index=False)
print("Saved cleaned_features.csv with shape", dataset4.drop(columns=["composition"]).shape)
