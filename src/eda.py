import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from prettytable import PrettyTable
from scipy import stats
from sklearn.preprocessing import LabelEncoder

from src.features import (
    categorical_features,
    continuous_features,
    new_categorical_features,
    new_continuous_features,
)

categorical_features = categorical_features + new_categorical_features
continuous_features = continuous_features + new_continuous_features

#####################
# Main EDA function #
#####################


def eda(
    df: pd.DataFrame,
    sale_col: str,
    revenue_col: str,
    continuous_features: list[str] = continuous_features,
    categorical_features: list[str] = categorical_features,
    exclude_features: list[str] = [],
    exclude_value: float | int = -1,
):
    logger.info("Performing EDA for Sale")
    eda_sales(df, sale_col)
    logger.info("Performing EDA for Sale-related Features")
    eda_sales_features(
        df,
        continuous_features,
        categorical_features,
        sale_col,
        exclude_features,
        exclude_value,
    )
    logger.info("Performing EDA for Revenue")
    eda_revenue(df[df[sale_col] == 1], revenue_col)
    logger.info("Performing EDA for Revenue-related Features")
    eda_revenue_features(
        df[df[sale_col] == 1], continuous_features, categorical_features, revenue_col
    )
    logger.info("Performing Continuous Features Correlation Analysis")
    analyze_continuous_feature_correlations(df, continuous_features)
    logger.info("Performing Categorical vs Categorical Correlation Analysis")
    analyze_categorical_feature_correlations(df, categorical_features)
    logger.info("Performing Categorical vs Continuous Correlation Analysis")
    analyze_categorical_vs_continuous_correlations(
        df, categorical_features, continuous_features
    )


########################
# Helper EDA functions #
########################


def print_table(data):
    table = PrettyTable()
    for key, value in data.items():
        table.add_column(key, [value])
    logger.info("\n" + table.get_string())


def analyze_continuous_feature_correlations(
    df: pd.DataFrame, continuous_features: list[str]
) -> None:
    corr_matrix = df[continuous_features].corr(method="spearman")
    plt.figure(figsize=(18, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".1f", cmap="coolwarm", vmin=-1, vmax=1, mask=mask
    )
    plt.title("Continuous Features Correlation (Spearman correlation)")
    plt.tight_layout()
    plt.show()


def analyze_continuous_vs_binary(
    df: pd.DataFrame, continuous_features: list[str], target_col: str
) -> None:
    cont_target_corr = pd.Series(index=continuous_features, dtype=float)
    le = LabelEncoder()
    encoded_target = le.fit_transform(df[target_col])
    for cont_col in continuous_features:
        correlation, _ = stats.pointbiserialr(encoded_target, df[cont_col])
        cont_target_corr[cont_col] = correlation
    plt.figure(figsize=(12, 6))
    cont_target_corr.plot(kind="bar")
    plt.title(f"Continuous Features vs {target_col}\n(Point-biserial correlation)")
    plt.tight_layout()
    plt.show()


def analyze_categorical_vs_target(
    df: pd.DataFrame, categorical_features: list[str], target_col: str
) -> None:
    cat_target_corr = pd.Series(index=categorical_features, dtype=float)
    for cat_col in categorical_features:
        contingency = pd.crosstab(df[cat_col], df[target_col])
        chi2, *_ = stats.chi2_contingency(contingency)
        n = len(df)
        min_dim = min(len(df[cat_col].unique()) - 1, len(df[target_col].unique()) - 1)
        if min_dim > 0:
            cramers_v = np.sqrt(chi2 / (n * min_dim))
        else:
            cramers_v = 0
        cat_target_corr[cat_col] = cramers_v
    plt.figure(figsize=(12, 6))
    cat_target_corr.plot(kind="bar")
    plt.title(f"Categorical Features vs {target_col}\n(Cramér's V)")
    plt.tight_layout()
    plt.show()


def analyze_continuous_vs_continuous(
    df: pd.DataFrame, continuous_features: list[str], target_col: str
) -> None:
    correlations = []
    for feature in continuous_features:
        correlation, _ = stats.spearmanr(df[feature], df[target_col])
        correlations.append(correlation)
    correlation_df = pd.Series(correlations, index=continuous_features)
    plt.figure(figsize=(12, 6))
    correlation_df.plot(kind="bar")
    plt.title(f"Continuous Features vs {target_col}\n(Spearman correlation)")
    plt.tight_layout()
    plt.show()


def analyze_categorical_feature_correlations(
    df: pd.DataFrame, categorical_features: list[str]
) -> None:
    results = {}
    for i, cat_col1 in enumerate(categorical_features):
        for j, cat_col2 in enumerate(categorical_features):
            if i < j:
                contingency = pd.crosstab(df[cat_col1], df[cat_col2])
                chi2, *_ = stats.chi2_contingency(contingency)
                n = len(df)
                min_dim = min(
                    len(df[cat_col1].unique()) - 1, len(df[cat_col2].unique()) - 1
                )
                cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
                results[(cat_col1, cat_col2)] = cramers_v
                results[(cat_col2, cat_col1)] = cramers_v
            elif i == j:
                results[(cat_col1, cat_col2)] = 1.0
    corr_matrix = pd.DataFrame(
        index=categorical_features,
        columns=categorical_features,
        data=[
            [results.get((i, j), 0) for j in categorical_features]
            for i in categorical_features
        ],
    )
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        fmt=".1f",
        mask=mask,
    )
    plt.title("Categorical Features vs Categorical Features\n(Cramér's V)")
    plt.tight_layout()
    plt.show()


def analyze_categorical_vs_continuous_correlations(
    df: pd.DataFrame, categorical_features: list[str], continuous_features: list[str]
) -> None:
    results = {}
    for cat_col in categorical_features:
        for cont_col in continuous_features:
            contingency = pd.crosstab(df[cat_col], pd.cut(df[cont_col], bins=10))
            chi2, *_ = stats.chi2_contingency(contingency)
            n = len(df)
            min_dim = min(len(df[cat_col].unique()) - 1, 10 - 1)
            cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
            results[(cat_col, cont_col)] = cramers_v

    corr_matrix = pd.DataFrame(
        index=categorical_features,
        columns=continuous_features,
        data=[
            [results.get((i, j), 0) for j in continuous_features]
            for i in categorical_features
        ],
    )
    plt.figure(figsize=(len(continuous_features), len(categorical_features)))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=0, vmax=1, fmt=".1f")
    plt.title("Categorical Features vs Continuous Features\n(Cramér's V)")
    plt.tight_layout()
    plt.show()


def eda_sales(df: pd.DataFrame, sale_col: str):
    logger.info("EDA for Sale Target")
    sale_counts = df[sale_col].value_counts(normalize=True)
    print_table(
        {
            "Sale Value": sale_counts.index.tolist(),
            "Percentage": sale_counts.values.tolist(),
        }
    )
    plt.figure(figsize=(6, 4))
    sns.countplot(x=sale_col, data=df)
    plt.title("Distribution of Sale")
    plt.show()


def eda_revenue(df: pd.DataFrame, revenue_col: str):
    logger.info("EDA for Revenue Target")
    revenue_stats = df[revenue_col].describe()
    print_table(revenue_stats.to_dict())
    plt.figure(figsize=(6, 4))
    sns.histplot(df[revenue_col], kde=True, bins=30)
    plt.title("Distribution of Revenue")
    plt.show()


def eda_sales_features(
    df: pd.DataFrame,
    continuous_features: list[str],
    categorical_features: list[str],
    sale_col: str,
    exclude_features: list[str] = [],
    exclude_value: float | int = -1,
):
    for feature in continuous_features:
        logger.info(f"EDA for Continuous Feature (Sale): {feature}")
        filtered_df = df.copy()
        if feature in exclude_features and feature in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[feature] != float(exclude_value)]
        fig, axes = plt.subplots(2, 2, figsize=(18, 8))
        sns.histplot(df[feature], kde=True, ax=axes[0, 0], bins=30)
        axes[0, 0].set_title(f"Distribution of {feature}")
        sns.boxplot(x=sale_col, y=feature, data=df, ax=axes[0, 1])
        axes[0, 1].set_title(f"Boxplot of {feature} by {sale_col}")
        if feature in exclude_features and feature in filtered_df.columns:
            sns.histplot(filtered_df[feature], kde=True, ax=axes[1, 0], bins=30)
            axes[1, 0].set_title(f"Filtered Distribution of {feature}")
            sns.boxplot(x=sale_col, y=feature, data=filtered_df, ax=axes[1, 1])
            axes[1, 1].set_title(f"Filtered Boxplot of {feature} by {sale_col}")
        else:
            axes[1, 0].set_visible(False)
            axes[1, 1].set_visible(False)
        plt.tight_layout()
        plt.show()

    for feature in categorical_features:
        logger.info(f"EDA for Categorical Feature (Sale): {feature}")
        filtered_df = df.copy()
        if feature in exclude_features and feature in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[feature] != float(exclude_value)]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.countplot(x=feature, data=df, hue=sale_col, ax=axes[0])
        axes[0].set_title(f"Distribution of {feature} by {sale_col}")
        if feature in exclude_features and feature in filtered_df.columns:
            sns.countplot(x=feature, data=filtered_df, hue=sale_col, ax=axes[1])
            axes[1].set_title(f"Filtered Distribution of {feature} by {sale_col}")
        plt.tight_layout()
        plt.show()

    logger.info("Analyzing correlations for sales features")
    analyze_continuous_vs_binary(df, continuous_features, sale_col)
    analyze_categorical_vs_target(df, categorical_features, sale_col)


def eda_revenue_features(
    df: pd.DataFrame,
    continuous_features: list[str],
    categorical_features: list[str],
    revenue_col: str,
):
    logger.info("Analyzing correlations for revenue features")
    analyze_continuous_vs_continuous(df, continuous_features, revenue_col)
    analyze_categorical_vs_target(df, categorical_features, revenue_col)


def eda_pairplot(df: pd.DataFrame, columns: list[str]):
    logger.info("Creating Pairplot")
    sns.pairplot(df[columns])
    plt.title("Pairplot")
    plt.show()
