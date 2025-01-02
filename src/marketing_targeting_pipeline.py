from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import copy

from src.features import (
    sale_cc_features,
    sale_cl_features,
    sale_mf_features,
    revenue_cc_features,
    revenue_cl_features,
    revenue_mf_features,
)
from src.dataframe_prep import train_test_split_and_preprocess
from src.model_training import nested_cv_optuna


#################
# Main pipeline #
#################


def full_pipeline(
    raw_excel_file: str,
    classifiers: list[tuple[callable, callable]],
    regressors: list[tuple[callable, callable]],
    split_params: dict[str, float | int] = None,
) -> tuple[dict[str, dict], pd.DataFrame]:
    """
    Executes the full pipeline for training classifiers and regressors.

    Args:
        raw_excel_file (str): Path to the raw Excel file.
        classifiers (list[tuple[callable, callable]]): List of tuples containing classifiers and their optimisation
            parameter functions.
        regressors (list[tuple[callable, callable]]): List of tuples containing regressors and their optimisation
            parameter functions.
        split_params (dict[str, float | int] | None): Parameters for train-test splitting. Defaults to None.

    Returns:
        tuple: Contains results for all trained models and their performance metrics, and the augmented test set.
    """
    split_params = split_params or {}
    datasets, test_set = train_test_split_and_preprocess(raw_excel_file)
    results = {}

    for suffix, dataset in datasets.items():
        if classifiers:
            results[f"Sale_{suffix}"] = copy.deepcopy(
                train_model_group(
                    dataset,
                    classifiers,
                    suffix,
                    target_type="Sale",
                    split_params=split_params,
                )
            )
        else:
            logger.info("No classifiers provided; skipping Sale_* models.")

        if regressors:
            results[f"Revenue_{suffix}"] = copy.deepcopy(
                train_model_group(
                    dataset,
                    regressors,
                    suffix,
                    target_type="Revenue",
                    split_params=split_params,
                )
            )
        else:
            logger.info("No regressors provided; skipping Revenue_* models.")

    results = retrain_best_models(results, datasets)
    test_set = augment_test_set(test_set, results)
    test_set = add_max_expected_revenue_column(test_set)
    test_set = assign_offers_and_rank(test_set)
    generate_report(test_set)
    return results, test_set


####################
# Helper Functions #
####################


def split_and_scale(
    data: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Preprocesses the data by splitting into train and test sets and applying standard scaling.

    Args:
        data (pd.DataFrame): The input dataframe.
        target_column (str): The name of the target column.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int, optional): Random state for reproducibility. Defaults to 42.
        stratify (pd.Series, optional): Data to use for stratification. Defaults to None.

    Returns:
        tuple: Scaled train and test feature matrices (X_train, X_test) and target arrays (y_train, y_test).
    """
    X = data.drop(columns=[target_column]).to_numpy()
    y = data[target_column].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def train_model_group(
    dataset: tuple[pd.DataFrame, pd.DataFrame],
    model_group: list[tuple[callable, callable]],
    suffix: str,
    target_type: str,
    split_params: dict[str, float | int],
) -> dict:
    """
    Trains a group of models for a specific target type and dataset.

    Args:
        dataset (tuple): Tuple containing the feature and target datasets.
        model_group (list): List of model and optimisation parameter function pairs.
        suffix (str): Identifier for the dataset (e.g., "MF", "CC").
        target_type (str): Type of target (e.g., "Sale" or "Revenue").
        split_params (dict): Parameters for train-test splitting.

    Returns:
        dict: Contains the best model, its parameters, scores, and test set performance.
    """
    logger.info(f"Training {target_type} models for {suffix}...")
    X_train, X_test, y_train, y_test = split_and_scale(
        dataset[0] if target_type == "Sale" else dataset[1],
        f"{target_type}_{suffix}",
        test_size=split_params.get("test_size", 0.2),
        random_state=split_params.get("random_state", 42),
        stratify=(
            dataset[0][f"{target_type}_{suffix}"] if target_type == "Sale" else None
        ),
    )
    best_model = None
    best_score = -float("inf") if target_type == "Sale" else float("inf")
    best_params = {}

    for model, optimise_params in model_group:
        model_name = type(model).__name__
        logger.info(f"Running nested CV for {model_name} on {target_type}_{suffix}...")
        try:
            cv_results = nested_cv_optuna(
                X_train, y_train, model, optimise_params, f"{target_type}_{suffix}"
            )
            best_result = (
                max(cv_results, key=lambda x: x["best_score"])
                if target_type == "Sale"
                else min(cv_results, key=lambda x: x["best_score"])
            )
            if (target_type == "Sale" and best_result["best_score"] > best_score) or (
                target_type == "Revenue" and best_result["best_score"] < best_score
            ):
                best_score = best_result["best_score"]
                best_model = model
                best_params = best_result["params"]
            logger.info(
                f"Best nested CV score for {model_name} on {target_type}_{suffix}: {best_score}"
            )
        except Exception as e:
            logger.info(f"Error with model {model_name} on {target_type}_{suffix}: {e}")

    if best_model is not None:
        best_model.set_params(**best_params)
        best_model.fit(X_train, y_train)
        if target_type == "Sale":
            y_train_prob = best_model.predict_proba(X_train)[:, 1]
            train_score = roc_auc_score(y_train, y_train_prob)
            y_test_prob = best_model.predict_proba(X_test)[:, 1]
            test_score = roc_auc_score(y_test, y_test_prob)
            logger.info(
                f"Train ROC AUC: {train_score}, Test ROC AUC: {test_score} "
                f"for {type(best_model).__name__} on {target_type}_{suffix}"
            )
        else:
            y_train_pred = best_model.predict(X_train)
            train_score = mean_squared_error(y_train, y_train_pred)
            y_test_pred = best_model.predict(X_test)
            test_score = mean_squared_error(y_test, y_test_pred)
            logger.info(
                f"Train MSE: {train_score}, Test MSE: {test_score} "
                f"for {type(best_model).__name__} on {target_type}_{suffix}"
            )
    else:
        train_score = None
        test_score = None
        logger.info(f"No valid model found for {target_type}_{suffix}.")
    return {
        "best_model": best_model,
        "best_cv_score": best_score,
        "best_params": best_params,
        "train_score": train_score,
        "test_score": test_score,
    }


def retrain_best_models(
    results: dict[str, dict],
    datasets: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> dict[str, dict]:
    """
    Retrains the best models on the full training dataset and evaluates their performance.

    Args:
        results (dict[str, dict]): Dictionary containing model results and configurations.
        datasets (dict[str, tuple[pd.DataFrame, pd.DataFrame]]): Dictionary containing datasets for retraining.

    Returns:
        dict[str, dict]: Updated results with retrained models and performance metrics.
    """
    for suffix, dataset in datasets.items():
        for target_type in ["Sale", "Revenue"]:
            key = f"{target_type}_{suffix}"
            if key in results and results[key]["best_model"] is not None:
                model = results[key]["best_model"]
                features = dataset[0] if target_type == "Sale" else dataset[1]
                target_column = f"{target_type}_{suffix}"

                X = features.drop(columns=[target_column])
                y = features[target_column]

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                logger.info(f"Retraining {key} model on full dataset...")
                model.fit(X_scaled, y)

                if target_type == "Sale":
                    y_prob = model.predict_proba(X_scaled)[:, 1]
                    metric = roc_auc_score(y, y_prob)
                    score = f"AUC: {metric}"
                else:
                    y_pred = model.predict(X_scaled)
                    metric = mean_squared_error(y, y_pred)
                    score = f"MSE: {metric}"

                logger.info(f"Retraining {key} model completed. {score}")

                results[key]["retrained_model"] = model
                results[key]["retrained_model_score"] = metric
                results[key]["scaler"] = scaler

    return results


def augment_test_set(test_set: pd.DataFrame, results: dict[str, dict]) -> pd.DataFrame:
    """
    Augments the test set with probabilities, predicted revenues, and expected revenues.

    Args:
        test_set (pd.DataFrame): Original test set.
        results (dict): Dictionary of trained models and their performance.

    Returns:
        pd.DataFrame: Test set augmented with new columns for probabilities, revenues, and expected revenues.
    """
    new_probability_columns = []
    new_revenue_columns = []

    for key, result in results.items():
        if result is not None:
            model = result["retrained_model"]
            scaler = result["scaler"]
            target_type, suffix = key.split("_")

            feature_set = get_feature_set(target_type, suffix)
            scaled_features = scaler.transform(test_set[feature_set])

            if target_type == "Sale":
                probabilities = model.predict_proba(scaled_features)[:, 1]
                test_set[f"{key}_probability"] = probabilities
                new_probability_columns.append(f"{key}_probability")
            else:
                predictions = model.predict(scaled_features)
                test_set[f"{key}_prediction"] = predictions
                new_revenue_columns.append(f"{key}_prediction")

    for probability_column in new_probability_columns:
        suffix = probability_column.split("_")[1]
        revenue_column = f"Revenue_{suffix}_prediction"
        if revenue_column in new_revenue_columns:
            test_set[f"Revenue_{suffix}_expected"] = (
                test_set[probability_column] * test_set[revenue_column]
            )

    return test_set


def add_max_expected_revenue_column(test_set: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a column to the test set indicating the maximum expected revenue across all products.

    Args:
        test_set (pd.DataFrame): The augmented test set with expected revenue columns.

    Returns:
        pd.DataFrame: Test set with a new column for maximum expected revenue.
    """
    expected_columns = [col for col in test_set.columns if col.endswith("_expected")]
    test_set["Max_Expected_Revenue"] = test_set[expected_columns].max(axis=1)
    return test_set


def assign_offers_and_rank(test_set: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns the optimal offer to each client based on maximum expected revenue and ranks clients.

    Args:
        test_set (pd.DataFrame): Test set with maximum expected revenue.

    Returns:
        pd.DataFrame: Test set with assigned offers and ranks.
    """
    expected_columns = [col for col in test_set.columns if col.endswith("_expected")]
    test_set["Assigned_Product"] = (
        test_set[expected_columns].idxmax(axis=1).str.split("_").str[1]
    )
    test_set = test_set.sort_values(
        by="Max_Expected_Revenue", ascending=False
    ).reset_index(drop=True)
    test_set["Target_Rank"] = range(1, len(test_set) + 1)
    return test_set


def generate_report(test_set: pd.DataFrame) -> None:
    """
    Generates a report of assigned offers, revenue, and visualizations.

    Args:
        test_set (pd.DataFrame): Test set with assigned offers and ranks.

    Returns:
        None
    """
    top_15_percent = int(len(test_set) * 0.15)
    selected_clients = test_set.head(top_15_percent)

    total_revenue = selected_clients["Max_Expected_Revenue"].sum()
    product_counts = selected_clients["Assigned_Product"].value_counts()

    logger.info(f"Total Expected Revenue: ${total_revenue:.2f}")
    for product, count in product_counts.items():
        logger.info(f"{product}: {count} clients")

    plt.figure(figsize=(10, 6))
    product_counts.plot(
        kind="bar",
        title="Product Assignments",
        xlabel="Product",
        ylabel="Number of Clients",
    )
    plt.savefig("product_assignments.png")
    logger.info("Saved bar chart: product_assignments.png")

    plt.figure(figsize=(8, 8))
    selected_clients["Assigned_Product"].value_counts().plot(
        kind="pie", autopct="%1.1f%%", title="Product Distribution"
    )
    plt.savefig("product_distribution.png")
    logger.info("Saved pie chart: product_distribution.png")


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
