import pandas as pd
import numpy as np
from src.features import (
    io_features,
    prod_bal_features,
    dem_features,
    continuous_features,
    categorical_features,
    asset_prods_suffix,
    liability_prods_suffix,
)

###############################
# Main preprocessing function #
###############################


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all preprocessing steps to the dataset.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    for column in dem_features:
        df = impute_random_proportion(df, column=column, random_state=42)
    for column in prod_bal_features:
        df = impute_with_constant(df, column=column, value=-1)
    for column in io_features:
        df = impute_with_constant(df, column=column, value=0)
    df = encode_sex(df, column="Sex")
    df = filter_by_age(df, column="Age", min_age=18)
    df = exclude_invalid_tenure(df, age_column="Age", tenure_column="Tenure")
    df = categorize_investors(df, column="Count_MF")
    df = create_new_features(df)
    df = adjust_dtypes(df)
    return df


###########################
# Handling missing values #
###########################


def impute_random_proportion(
    df: pd.DataFrame, column: str, random_state: int = None
) -> pd.DataFrame:
    """
    Impute missing values in a column based on the observed proportion of categories.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column to impute.
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with imputed values.
    """
    df = df.copy()
    rng = np.random.default_rng(random_state)
    proportions = df[column].value_counts(normalize=True)
    missing_mask = df[column].isna()
    if missing_mask.any():
        df.loc[missing_mask, column] = rng.choice(
            proportions.index, size=missing_mask.sum(), p=proportions.values
        )
    return df


def impute_with_constant(df: pd.DataFrame, column: str, value: float) -> pd.DataFrame:
    """
    Impute missing values in a column with a constant value.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column to impute.
        value (float): Constant value to impute.

    Returns:
        pd.DataFrame: DataFrame with imputed values.
    """
    df = df.copy()
    df.loc[:, column] = df[column].fillna(value)
    return df


#######################
# Encoding sex column #
#######################


def encode_sex(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Encode the Sex column: 1 for Male, 0 for Female.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column representing Sex.

    Returns:
        pd.DataFrame: DataFrame with encoded Sex column.
    """
    df = df.copy()
    df[column] = df[column].map({"M": 1, "F": 0})
    return df


#################
# Sanity checks #
#################


def filter_by_age(df: pd.DataFrame, column: str, min_age: int) -> pd.DataFrame:
    """
    Filter out rows where age is less than the specified minimum.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Age column to check.
        min_age (int): Minimum age.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    return df[df[column] >= min_age]


def exclude_invalid_tenure(
    df: pd.DataFrame, age_column: str, tenure_column: str
) -> pd.DataFrame:
    """
    Exclude rows where Age <= Tenure, assuming the account was inherited.

    Args:
        df (pd.DataFrame): Input DataFrame.
        age_column (str): Column representing age.
        tenure_column (str): Column representing tenure.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    return df[df[age_column] * 12 > df[tenure_column]]


###############################
# Reducing feature categories #
###############################


def categorize_investors(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Categorize investors based on the number of mutual funds they own.
    Includes a category for missing values (-1).

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column representing the count of mutual funds.

    Returns:
        pd.DataFrame: DataFrame with categorized investor types.
    """
    df = df.copy()
    bins = [-2, 0, 3, 10, np.inf]
    labels = [-1, 1, 2, 3]
    df[column] = pd.cut(df[column], bins=bins, labels=labels, right=True).astype("int")
    return df


##########################
# Adjust data types step #
##########################


def adjust_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust data types for categorical and continuous features.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with updated data types.
    """
    df[categorical_features] = df[categorical_features].astype(int)
    df[continuous_features] = df[continuous_features].astype(float)
    return df


##########################
# Feature creation steps #
##########################


def create_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features based on existing columns, such as aggregations and ratios.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with additional features.
    """
    df = df.copy()
    df = create_total_asset_count(df)
    df = create_total_liability_count(df)
    df = create_actbal_total(df)
    df = create_debt_to_balance(df)
    df = create_inflow_outflow_ratio(df)
    return df


#####################################
# Feature creation helper functions #
#####################################


def create_total_asset_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a feature representing the total count of assets.
    Ensures -1 values (missing) are handled appropriately to avoid unrealistic totals.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with Count_Total_Asset feature.
    """
    df = df.copy()
    count_features_asset = [f"Count_{suffix}" for suffix in asset_prods_suffix]
    df["Count_Total_Asset"] = df[count_features_asset].clip(lower=0).sum(axis=1)
    return df


def create_total_liability_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a feature representing the total count of liabilities.
    Ensures values are adjusted such that 0 becomes -1 if no liabilities exist.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with Count_Total_Liability feature.
    """
    df = df.copy()
    count_features_liability = [f"Count_{suffix}" for suffix in liability_prods_suffix]
    df["Count_Total_Liability"] = df[count_features_liability].clip(lower=0).sum(axis=1)
    df.loc[df["Count_Total_Liability"] == 0, "Count_Total_Liability"] = -1
    return df


def create_actbal_total(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a feature representing the total account balance (assets - liabilities).

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with ActBal_Total feature.
    """
    df = df.copy()
    actbal_features_asset = [f"ActBal_{suffix}" for suffix in asset_prods_suffix]
    actbal_features_liability = [
        f"ActBal_{suffix}" for suffix in liability_prods_suffix
    ]
    df["ActBal_Total_Asset"] = df[actbal_features_asset].clip(lower=0).sum(axis=1)
    df["ActBal_Total_Liability"] = (
        df[actbal_features_liability].clip(lower=0).sum(axis=1)
    )
    df["ActBal_Total"] = df["ActBal_Total_Asset"] - df["ActBal_Total_Liability"]
    return df


def create_debt_to_balance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a feature representing the debt-to-balance ratio.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with Debt_To_Balance feature.
    """
    df = df.copy()
    df["Debt_To_Balance"] = df["ActBal_Total_Liability"] / (df["ActBal_Total"] + 1e-6)
    return df


def create_inflow_outflow_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a feature representing the debt-to-income ratio.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with Inflow_Outflow_Ratio feature.
    """
    df = df.copy()
    df["Inflow_Outflow_Ratio"] = df["VolumeCred"] / (df["VolumeDeb"] + 1e-6)
    return df
