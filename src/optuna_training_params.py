import optuna


def optimise_params_logistic_regression(trial):
    """Define the hyperparameter search space for Logistic Regression with valid constraints."""
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", None])
    solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear", "saga"])

    # Ensure valid combinations of penalty and solver
    if penalty == "l1" and solver not in ["liblinear", "saga"]:
        raise optuna.exceptions.TrialPruned()
    if penalty == "elasticnet" and solver != "saga":
        raise optuna.exceptions.TrialPruned()
    if penalty is None and solver != "lbfgs":
        raise optuna.exceptions.TrialPruned()

    l1_ratio = None
    if penalty == "elasticnet":
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)

    return {
        "penalty": penalty,
        "C": trial.suggest_float("C", 0.001, 10, log=True),
        "solver": solver,
        "max_iter": trial.suggest_int("max_iter", 100, 1000),
        "l1_ratio": l1_ratio,
    }


def optimise_params_random_forest_classifier(trial):
    """Define the hyperparameter search space for Random Forest Classifier."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 5, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }


def optimise_params_ridge_regressor(trial):
    """Define the hyperparameter search space for Ridge Regression."""
    return {
        "alpha": trial.suggest_float("alpha", 0.01, 10.0, log=True),
        "solver": trial.suggest_categorical(
            "solver", ["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"]
        ),
    }


def optimise_params_random_forest_regressor(trial):
    """Define the hyperparameter search space for Random Forest Regressor."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }
