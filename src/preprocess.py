"""
preprocess.py
Data loading, cleaning, and feature engineering for bike demand modeling.
"""

import numpy as np
import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """Load the daily bike-sharing dataset from a CSV file."""
    return pd.read_csv(filepath)


def engineer_features(day: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps to the daily dataset.

    Adds:
    - log_cnt: log-transformed response variable
    - temp_sq: squared temperature for quadratic models
    - Categorical dtypes for season, yr, workingday, weathersit
    - Human-readable label columns for season and weathersit
    - season_dummies and temperature-season interaction columns for CV models
    """
    day = day.copy()

    # Transformed response and quadratic predictor
    day["log_cnt"] = np.log(day["cnt"])
    day["temp_sq"] = day["temp"] ** 2

    # Categorical variables
    cat_vars = ["season", "yr", "workingday", "weathersit"]
    for col in cat_vars:
        day[col] = day[col].astype("category")

    # Human-readable labels
    season_labels = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}
    weather_labels = {
        1: "Clear",
        2: "Mist/Cloudy",
        3: "Light Rain/Snow",
        4: "Heavy Rain/Snow",
    }
    day["season_label"] = day["season"].astype(int).map(season_labels)
    day["weather_label"] = day["weathersit"].astype(int).map(weather_labels)

    day["season_label"] = pd.Categorical(
        day["season_label"],
        categories=["Spring", "Summer", "Fall", "Winter"],
        ordered=True,
    )
    day["weather_label"] = pd.Categorical(
        day["weather_label"],
        categories=["Clear", "Mist/Cloudy", "Light Rain/Snow", "Heavy Rain/Snow"],
        ordered=True,
    )

    return day


def add_interaction_features(day: pd.DataFrame) -> pd.DataFrame:
    """
    Add season dummy variables and temperature-season interaction terms.
    Returns an augmented copy of the dataframe (used for cross-validation
    of Model 4).
    """
    day_cv = day.copy()
    season_dummies = pd.get_dummies(day_cv["season"], prefix="season", drop_first=True)
    day_cv = pd.concat([day_cv, season_dummies], axis=1)

    for col in season_dummies.columns:
        day_cv[f"temp_{col}"] = day_cv["temp"] * day_cv[col]

    return day_cv


def load_and_prepare(filepath: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function: load, engineer features, and return both the
    base dataframe and the CV-augmented dataframe.
    """
    day = load_data(filepath)
    day = engineer_features(day)
    day_cv = add_interaction_features(day)
    return day, day_cv
