import mlflow
import numpy as np
import optuna
from loguru import logger
from optuna.samplers import TPESampler


from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, mean_squared_error, roc_auc_score
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold


sampler = TPESampler(seed=42)
trials = 50
n_splits_outer = 5
n_splits_inner = 3

##########################
# Main training function #
##########################


def nested_cv_optuna(
    X: np.ndarray,
    y: np.ndarray,
    model: BaseEstimator,
    optimise_params: callable,
    dataset_name: str,
    n_splits_outer: int = n_splits_outer,
    n_splits_inner: int = n_splits_inner,
    n_trials: int = trials,
    n_jobs: int = None,
    random_state: int = 42,
) -> list[dict]:
    """
    Performs nested cross-validation with Optuna hyperparameter optimisation.

    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        model (BaseEstimator): Base model for training
        optimise_params (callable): Function defining hyperparameter search space
        dataset_name (str): Name of the dataset
        n_splits_outer (int): Number of outer CV splits
        n_splits_inner (int): Number of inner CV splits
        n_trials (int): Number of Optuna trials
        n_jobs (int): Number of parallel jobs
        random_state (int): Random state for reproducibility

    Returns:
        list[dict]: Results containing optimised models and their metrics

    Raises:
        ValueError: If input parameters are invalid
        OSError: If there are file operation errors
    """
    logger.info(f"Starting nested CV with Optuna for dataset: {dataset_name}")

    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have same number of samples. Got {X.shape[0]} and {y.shape[0]}"
        )

    is_classification = hasattr(model, "predict_proba") or len(set(y)) <= 2
    outer_cv, inner_cv = setup_cross_validators(
        is_classification, n_splits_outer, n_splits_inner, random_state
    )

    with mlflow.start_run(run_name=f"{type(model).__name__}_{dataset_name}"):
        mlflow.log_params(
            {
                "n_splits_outer": n_splits_outer,
                "n_splits_inner": n_splits_inner,
                "n_trials": n_trials,
                "n_jobs": n_jobs,
                "random_state": random_state,
            }
        )
        all_best_scores = []
        results = []

        for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            with mlflow.start_run(run_name=f"fold_{i+1}", nested=True):
                objective = create_objective_function(
                    model,
                    X_train,
                    y_train,
                    optimise_params,
                    inner_cv,
                    is_classification,
                    n_jobs,
                )
                best_params, best_score = optimise_hyperparameters(
                    objective, n_trials, is_classification
                )
                mlflow.log_params(best_params)
                mlflow.log_metric("nested_cv_best_score", best_score)
                fold_results = evaluate_fold(
                    model,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    best_params,
                    is_classification,
                )
                results.append(fold_results)
                all_best_scores.append(fold_results["best_score"])
        mlflow.log_metric("average_nested_cv_score", np.mean(all_best_scores))

    return results


#############################
# Training helper functions #
#############################


def setup_cross_validators(
    is_classification: bool,
    n_splits_outer: int,
    n_splits_inner: int,
    random_state: int,
) -> tuple[StratifiedKFold | KFold, StratifiedKFold | KFold]:
    """
    Sets up cross-validation objects for nested CV.

    Args:
        is_classification (bool): Whether the task is classification
        n_splits_outer (int): Number of outer CV splits
        n_splits_inner (int): Number of inner CV splits
        random_state (int): Random state for reproducibility

    Returns:
        tuple[StratifiedKFold | KFold, StratifiedKFold | KFold]: Outer and inner CV objects
    """
    cv_class = StratifiedKFold if is_classification else KFold
    outer_cv = cv_class(
        n_splits=n_splits_outer, shuffle=True, random_state=random_state
    )
    inner_cv = cv_class(
        n_splits=n_splits_inner, shuffle=True, random_state=random_state
    )

    return outer_cv, inner_cv


def create_objective_function(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    optimise_params: callable,
    inner_cv: StratifiedKFold | KFold,
    is_classification: bool,
    n_jobs: int,
) -> callable:
    """
    Creates the objective function for Optuna optimisation.

    Args:
        model (BaseEstimator): Model to optimise
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training targets
        optimise_params (callable): Function to get parameters from trial
        inner_cv (StratifiedKFold | KFold): Inner CV splitter
        is_classification (bool): Whether task is classification
        n_jobs (int): Number of parallel jobs

    Returns:
        callable: Objective function for optimisation
    """

    def objective(trial):
        params = optimise_params(trial)
        model.set_params(**params)
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=inner_cv,
            scoring="roc_auc" if is_classification else "neg_mean_squared_error",
            n_jobs=n_jobs,
        )
        return np.mean(scores) if is_classification else -np.mean(scores)

    return objective


def optimise_hyperparameters(
    objective: callable,
    n_trials: int,
    is_classification: bool,
) -> tuple[dict, float]:
    """
    Optimises hyperparameters using Optuna.

    Args:
        objective (callable): Objective function for optimisation
        n_trials (int): Number of optimisation trials
        is_classification (bool): Whether task is classification

    Returns:
        tuple[dict, float]: Best parameters and best score
    """
    direction = "maximize" if is_classification else "minimize"
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    return study.best_params, study.best_value


def evaluate_fold(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    best_params: dict,
    is_classification: bool,
) -> dict:
    """
    Evaluates model on a single fold.

    Args:
        model (BaseEstimator): Model to evaluate
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training targets
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test targets
        best_params (dict): Best parameters from optimisation
        is_classification (bool): Whether task is classification

    Returns:
        dict: Evaluation results
    """
    model.set_params(**best_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if is_classification:
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )
        score = log_full_classification_report(
            y_test, y_pred, y_prob, context="nested_cv"
        )
    else:
        score = log_regression_metrics(y_test, y_pred, context="nested_cv")

    return {
        "model": model,
        "params": best_params,
        "y_test": y_test,
        "y_pred": y_pred,
        "best_score": score,
    }


##################
# Metrics logger #
##################


def log_full_classification_report(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    context: str = "",
) -> float:
    """
    Logs a full classification report and AUROC.

    Args:
        y_test (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        y_prob (np.ndarray | None): Predicted probabilities.
        context (str): Additional context for logging (e.g., "nested_cv" or "test_data").

    Returns:
        float: AUROC score if probabilities are provided, otherwise 0.0.

    """
    logger.info(f"Starting classification report logging for context: {context}")

    report = classification_report(y_test, y_pred, output_dict=True)

    for key, value in report.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                metric_name = f"{context}_{key}_{sub_key}"
                mlflow.log_metric(metric_name, sub_value)

    logger.info("Logged Classification report")

    if y_prob is not None:
        auroc = roc_auc_score(y_test, y_prob)
        mlflow.log_metric(f"{context}_auroc", auroc)
        logger.info(f"Logged AUROC score: {auroc}")
        return auroc

    return 0.0


def log_regression_metrics(
    y_test: np.ndarray, y_pred: np.ndarray, context: str = ""
) -> float:
    """
    Logs regression metrics including Mean Squared Error (MSE).

    Args:
        y_test (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
        context (str): Additional context for logging (e.g., "nested_cv" or "test_data").

    Returns:
        float: MSE of the predictions.

    """
    logger.info(f"Starting regression metrics logging for context: {context}")

    mse = mean_squared_error(y_test, y_pred)
    mlflow.log_metric(f"{context}_mean_squared_error", mse)
    logger.info(f"Logged MSE: {mse}")

    return mse
