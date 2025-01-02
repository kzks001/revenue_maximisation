import pandas as pd

from src.features import (
    dem_features,
    io_features,
    prod_bal_features,
    revenue_cc_features,
    revenue_cl_features,
    revenue_mf_features,
    revenue_target,
    sale_cc_features,
    sale_cl_features,
    sale_mf_features,
    sale_target,
    suffixes,
)
from src.file_path import raw_data_directory
from src.preprocessing import preprocess_data
from src.read_data import ingest_and_merge_data


def train_test_split_and_preprocess(
    raw_excel_file: str,
) -> tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]:
    """
    Processes a raw dataset, splits it into train-test sets, preprocesses the train set, and creates separate
    datasets for sales and revenue targets based on given suffixes.

    Args:
        raw_excel_file (str): Name of the raw Excel file. Must be in data/raw.

    Returns:
        tuple: A dictionary with keys as suffixes and values as tuples of DataFrames for sales and revenue data,
            and the test set DataFrame.
    """
    _, df = ingest_and_merge_data(raw_data_directory + raw_excel_file)
    features_columns = dem_features + prod_bal_features + io_features
    target_columns = sale_target + revenue_target
    test_set_mask = df[target_columns].isna().all(axis=1)
    train_set = df[~test_set_mask.to_numpy()].drop(columns="Client")
    test_set = df[test_set_mask][["Client"] + features_columns]
    df = preprocess_data(train_set)
    test_set = preprocess_data(test_set)
    result = {}

    for suffix in suffixes:
        sale_features = get_feature_set("sale", suffix)
        revenue_features = get_feature_set("revenue", suffix)
        result[suffix] = (
            df[sale_features + [f"Sale_{suffix}"]],
            df[df[f"Sale_{suffix}"] == 1][revenue_features + [f"Revenue_{suffix}"]],
        )

    return result, test_set


def get_feature_set(target_type: str, suffix: str) -> list[str]:
    """
    Safely retrieves the feature set for a given target type and suffix.

    Args:
        target_type (str): The target type, either "sale" or "revenue".
        suffix (str): The suffix identifying the feature set (e.g., "cc", "cl", "mf").

    Returns:
        list[str]: The feature set corresponding to the target type and suffix.
    """
    feature_sets = {
        "sale_cc": sale_cc_features,
        "sale_cl": sale_cl_features,
        "sale_mf": sale_mf_features,
        "revenue_cc": revenue_cc_features,
        "revenue_cl": revenue_cl_features,
        "revenue_mf": revenue_mf_features,
    }
    key = f"{target_type.lower()}_{suffix.lower()}"
    return feature_sets[key]
