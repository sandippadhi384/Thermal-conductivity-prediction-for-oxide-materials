
"""
Data curation and feature generation for oxide thermal-conductivity ML.

Workflow
--------
1. Load the oxide dataset
2. Convert chemical formulas to pymatgen Composition type formula for further processing
3. Generate composition-based features using matminer
4. change the categorical features to numerical using one-hot-encoder 
5. Remove missing, zero deviation, and duplicate features 

This code follows the feature extraction workflow used in this research work.
"""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from matminer.featurizers.composition import (
    ElementProperty,
    ElementFraction,
    Stoichiometry,
    AtomicOrbitals,
    YangSolidSolution,
    Meredig,
)
from matminer.featurizers.conversions import StrToComposition

RANDOM_STATE = 42

# Maximum allowed fraction of missing values in a feature column.
# Keep this consistent with the original notebook.
MISSING_THRESHOLD = 0.50

# Columns that should not be used as ML features.
TARGET_COLUMN = "thermal_conductivity"

FORMULA_COLUMN = "composition"


#load the data

def load_dataset(file_path: str | Path) -> pd.DataFrame:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path)
    elif file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Use .xlsx, .xls, or .csv.")
    print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df

# Data curation
#remove space if there and add "_"
def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.astype(str).str.strip().str.replace(" ", "_", regex=False))
    return df

#handle any missing values
def remove_high_missing_columns(df: pd.DataFrame,threshold: float = MISSING_THRESHOLD,) -> pd.DataFrame:
    df = df.copy()
    missing_fraction = df.isna().mean()

    columns_to_remove = missing_fraction[missing_fraction > threshold].index.tolist()

    if columns_to_remove:
        print(f"Removing {len(columns_to_remove)} columns "
            f"with > {threshold:.0%} missing values.")
        df = df.drop(columns=columns_to_remove)
    return df

# remove if any constant coloumns
def remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    constant_columns = [column for column in df.columns if df[column].nunique(dropna=False) <= 1]
    if constant_columns:
        print(f"Removing {len(constant_columns)} constant columns.")
        df = df.drop(columns=constant_columns)
    return df

#remove duplicate columns
def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    duplicated_columns = df.T.duplicated()
    columns_to_remove = df.columns[duplicated_columns].tolist()
    if columns_to_remove:
        print(f"Removing {len(columns_to_remove)} duplicate columns.")
        df = df.loc[:, ~duplicated_columns]
    return df

# data curation
def curate_dataset(
    df: pd.DataFrame,
    missing_threshold: float = MISSING_THRESHOLD) -> pd.DataFrame:
    print("\nStarting data curation...")
    df = standardize_column_names(df)
    print(f"Initial shape: {df.shape}")
    df = remove_high_missing_columns(df,threshold=missing_threshold)
    df = remove_constant_columns(df)
    df = remove_duplicate_columns(df)
    print(f"Final curated shape: {df.shape}")
    return df

# converting formulas into composition
def convert_formulas_to_compositions(
    df: pd.DataFrame,formula_column: str = FORMULA_COLUMN,) -> pd.DataFrame:
    df = df.copy()

    if formula_column not in df.columns:
        raise KeyError(
            f"'{formula_column}' column was not found in the dataset.")
    print("\nConverting chemical formulas to Composition objects...")
    converter = StrToComposition(target_col_id="composition_obj")
    df = converter.featurize_dataframe(df,formula_column,ignore_errors=True)
    return df


#features extraction using compositions
def generate_element_property_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    print("\nadding DEML elemental-property features...")
    deml = ElementProperty.from_preset("deml")
    df = deml.featurize_dataframe(df,col_id="composition_obj",ignore_errors=True)
    print("Generating Magpie elemental-property features...")
    magpie = ElementProperty.from_preset("magpie")
    df = magpie.featurize_dataframe(df,col_id="composition_obj",ignore_errors=True)
    return df
# elemental fraction features
def generate_element_fraction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    print("\nGenerating elemental fraction features...")
    featurizer = ElementFraction()
    df = featurizer.featurize_dataframe(df,col_id="composition_obj",ignore_errors=True)
    return df
# stociometric features
def generate_stoichiometry_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    print("\nGenerating stoichiometry features...")
    featurizer = Stoichiometry()
    df = featurizer.featurize_dataframe(df,col_id="composition_obj",ignore_errors=True)
    return df
# atomic oorbital features
def generate_atomic_orbital_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    print("\nGenerating atomic orbital features...")
    featurizer = AtomicOrbitals()
    df = featurizer.featurize_dataframe(df,col_id="composition_obj",ignore_errors=True)
    return df
# yang features
def generate_yang_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    print("\nGenerating Yang solid-solution features...")
    featurizer = YangSolidSolution()
    df = featurizer.featurize_dataframe(df,col_id="composition_obj",ignore_errors=True)
    return df
#meredig features
def generate_meredig_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    print("\nGenerating Meredig features...")
    featurizer = Meredig()
    df = featurizer.featurize_dataframe(df,col_id="composition_obj",ignore_errors=True)
    return df

#generate all features
def generate_composition_features(df: pd.DataFrame,) -> pd.DataFrame:
    print("COMPOSITION FEATURE GENERATION")
    df = generate_element_property_features(df)
    df = generate_element_fraction_features(df)
    df = generate_stoichiometry_features(df)
    df = generate_atomic_orbital_features(df)
    df = generate_yang_features(df)
    df = generate_meredig_features(df)
    print(f"\nFeature generation completed: "f"{df.shape[0]} rows × {df.shape[1]} columns")
    return df

# convert categorical features into numerial columns using one hot encoder (HOMO/LUMO features)
def encode_categorical_features(df: pd.DataFrame,categorical_columns: Iterable[str] | None = None) -> pd.DataFrame:
    df = df.copy()
    if categorical_columns is None:
        categorical_columns = [column for column in ["HOMO", "LUMO"] if column in df.columns]
    categorical_columns = list(categorical_columns)
    if not categorical_columns:
        print("\nNo categorical HOMO/LUMO columns found.")
        return df
    print(f"\nEncoding categorical columns:{categorical_columns}")
    df = pd.get_dummies(df,columns=categorical_columns,dtype=int,)
    return df


#feature clening
def remove_non_feature_columns(df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    formula_column: str = FORMULA_COLUMN) -> pd.DataFrame:
    df = df.copy()
    columns_to_drop = []
    # chemical formula is not an ML feature.
    if formula_column in df.columns:
        columns_to_drop.append(formula_column)
    # The pymatgen Composition object cannot be used for ML models.
    if "composition_obj" in df.columns:
        columns_to_drop.append("composition_obj")
    columns_to_drop = list(set(columns_to_drop))
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors="ignore")
    return df

# remove columns with missing values
def remove_feature_columns_with_missing_values(df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    threshold: float = MISSING_THRESHOLD) -> pd.DataFrame:
    df = df.copy()
    feature_columns = [column for column in df.columns if column != target_column]
    missing_fraction = df[feature_columns].isna().mean()
    columns_to_remove = missing_fraction[missing_fraction > threshold].index.tolist()
    if columns_to_remove:
        print(f"Removing {len(columns_to_remove)} feature columns "
            f"with excessive missing values.")
        df = df.drop(columns=columns_to_remove)
    return df

#remove contant features having zero deviation
def remove_constant_features(df: pd.DataFrame,target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    df = df.copy()
    feature_columns = [column for column in df.columns if column != target_column]
    constant_columns = [column for column in feature_columns if df[column].nunique(dropna=False) <= 1]
    if constant_columns:
        print(f"Removing {len(constant_columns)} constant features.")
        df = df.drop(columns=constant_columns)
    return df

# remove duplicate features
def remove_duplicate_features(df: pd.DataFrame,target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    df = df.copy()
    target = None
    if target_column in df.columns:
        target = df[target_column].copy()
        feature_df = df.drop(columns=[target_column])
    else:
        feature_df = df
    duplicated = feature_df.T.duplicated()
    duplicate_columns = feature_df.columns[duplicated].tolist()
    if duplicate_columns:
        print(f"Removing {len(duplicate_columns)} duplicate features.")
        feature_df = feature_df.loc[:,~duplicated]
    if target is not None:
        feature_df[target_column] = target
    return feature_df
#clean features
def clean_features(df: pd.DataFrame,target_column: str = TARGET_COLUMN,
    missing_threshold: float = MISSING_THRESHOLD) -> pd.DataFrame:
    print("FEATURE CLEANING")
    df = remove_non_feature_columns(df,target_column=target_column)
    df = remove_feature_columns_with_missing_values(df,target_column=target_column,
        threshold=missing_threshold)
    df = remove_constant_features(df,target_column=target_column)
    df = remove_duplicate_features(df,target_column=target_column)
    print(f"\nFinal feature dataset: "
        f"{df.shape[0]} rows × {df.shape[1]} columns")
    return df

#build the features and all
def build_feature_dataset(
    input_file: str | Path,
    output_file: str | Path,
    target_column: str = TARGET_COLUMN,
    formula_column: str = FORMULA_COLUMN,
    missing_threshold: float = MISSING_THRESHOLD) -> pd.DataFrame:
    #load data
    df = load_dataset(input_file)
    # Data curation
    df = curate_dataset(df,missing_threshold=missing_threshold)
    #Convert formulas to compositions
    df = convert_formulas_to_compositions(df,formula_column=formula_column)
    #generate composition-based features
    df = generate_composition_features(df)
    #use one hot encoder
    df = encode_categorical_features(df)
    #feature clening
    df = clean_features(df,target_column=target_column,
        missing_threshold=missing_threshold)
    #save the final output
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(output_file,index=False)
    if target_column in df.columns:
        print(f"Target      : {target_column}")

    return df

#grt the output after all operations
INPUT_FILE = "data/raw/Final ML Datasets - Copy.xlsx" #my file name
OUTPUT_FILE = "data/processed/oxide_features.csv"
build_feature_dataset(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        target_column="thermal_conductivity",
        formula_column="composition",
        missing_threshold=0.50)