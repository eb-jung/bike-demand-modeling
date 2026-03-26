"""
models.py
Model fitting, model specifications, and cross-validated RMSE for
bike demand modeling.
"""

import pandas as pd
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Model fitting

def fit_count_models(day: pd.DataFrame) -> dict:
    """
    Fit the four OLS models on raw cnt (count response).

    Returns a dict with keys: model1, model2, model3, model4.
    """
    model1 = smf.ols("cnt ~ temp + hum + windspeed", data=day).fit()

    model2 = smf.ols(
        "cnt ~ temp + hum + windspeed + C(season) + C(workingday) + C(yr)",
        data=day,
    ).fit()

    model3 = smf.ols(
        "cnt ~ temp + temp_sq + hum + windspeed + C(season) + C(workingday) + C(yr)",
        data=day,
    ).fit()

    model4 = smf.ols(
        "cnt ~ temp * C(season) + hum + windspeed + C(workingday) + C(yr)",
        data=day,
    ).fit()

    return {"model1": model1, "model2": model2, "model3": model3, "model4": model4}


def fit_log_models(day: pd.DataFrame) -> dict:
    """
    Fit the four OLS models on log_cnt (log-response).

    Returns a dict with keys: log_model1, log_model2, log_model3, log_model4.
    """
    log_model1 = smf.ols("log_cnt ~ temp + hum + windspeed", data=day).fit()

    log_model2 = smf.ols(
        "log_cnt ~ temp + hum + windspeed + C(season) + C(workingday) + C(yr)",
        data=day,
    ).fit()

    log_model3 = smf.ols(
        "log_cnt ~ temp + temp_sq + hum + windspeed + C(season) + C(workingday) + C(yr)",
        data=day,
    ).fit()

    log_model4 = smf.ols(
        "log_cnt ~ temp * C(season) + hum + windspeed + C(workingday) + C(yr)",
        data=day,
    ).fit()

    return {
        "log_model1": log_model1,
        "log_model2": log_model2,
        "log_model3": log_model3,
        "log_model4": log_model4,
    }


# Model comparison tables

def comparison_table_count(models: dict) -> pd.DataFrame:
    """Build a summary comparison table for the four count models."""
    from statsmodels.stats.stattools import durbin_watson

    m1, m2, m3, m4 = (
        models["model1"],
        models["model2"],
        models["model3"],
        models["model4"],
    )
    return pd.DataFrame(
        {
            "Model": [
                "Model 1: Weather only",
                "Model 2: Weather + calendar",
                "Model 3: Quadratic temperature",
                "Model 4: Temp x season interaction",
            ],
            "R_squared": [m1.rsquared, m2.rsquared, m3.rsquared, m4.rsquared],
            "Adj_R_squared": [
                m1.rsquared_adj,
                m2.rsquared_adj,
                m3.rsquared_adj,
                m4.rsquared_adj,
            ],
            "AIC": [m1.aic, m2.aic, m3.aic, m4.aic],
            "BIC": [m1.bic, m2.bic, m3.bic, m4.bic],
            "Durbin_Watson": [
                durbin_watson(m1.resid),
                durbin_watson(m2.resid),
                durbin_watson(m3.resid),
                durbin_watson(m4.resid),
            ],
        }
    )


def comparison_table_log(log_models: dict) -> pd.DataFrame:
    """Build a summary comparison table for the four log-response models."""
    from statsmodels.stats.stattools import durbin_watson

    lm1, lm2, lm3, lm4 = (
        log_models["log_model1"],
        log_models["log_model2"],
        log_models["log_model3"],
        log_models["log_model4"],
    )
    return pd.DataFrame(
        {
            "Model": ["Log_Model1", "Log_Model2", "Log_Model3", "Log_Model4"],
            "R_squared": [lm1.rsquared, lm2.rsquared, lm3.rsquared, lm4.rsquared],
            "Adj_R2": [
                lm1.rsquared_adj,
                lm2.rsquared_adj,
                lm3.rsquared_adj,
                lm4.rsquared_adj,
            ],
            "AIC": [lm1.aic, lm2.aic, lm3.aic, lm4.aic],
            "BIC": [lm1.bic, lm2.bic, lm3.bic, lm4.bic],
            "Durbin_Watson": [
                durbin_watson(lm1.resid),
                durbin_watson(lm2.resid),
                durbin_watson(lm3.resid),
                durbin_watson(lm4.resid),
            ],
        }
    )


# Cross-validation

def cv_rmse(
    df: pd.DataFrame,
    feature_cols: list,
    categorical_cols: list,
    response: str = "cnt",
    n_splits: int = 10,
) -> tuple[float, float]:
    """
    Compute 10-fold cross-validated RMSE for a linear regression specified
    by feature_cols (with categorical_cols one-hot encoded).

    Returns (mean_rmse, std_rmse).
    """
    X = df[feature_cols].copy()
    y = df[response].copy()

    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        [
            ("num", "passthrough", numeric_cols),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                categorical_cols,
            ),
        ]
    )

    pipeline = Pipeline([("preprocess", preprocessor), ("model", LinearRegression())])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_scores = -cross_val_score(
        pipeline, X, y, cv=kf, scoring="neg_root_mean_squared_error"
    )

    return rmse_scores.mean(), rmse_scores.std()


def run_cv_count(day: pd.DataFrame, day_cv: pd.DataFrame) -> pd.DataFrame:
    """
    Run 10-fold CV for all four count models and return a summary table.
    """
    cv_specs = {
        "Model 1: Weather only": {
            "features": ["temp", "hum", "windspeed"],
            "categorical": [],
        },
        "Model 2: Weather + calendar": {
            "features": ["temp", "hum", "windspeed", "season", "workingday", "yr"],
            "categorical": ["season", "workingday", "yr"],
        },
        "Model 3: Quadratic temperature": {
            "features": [
                "temp",
                "temp_sq",
                "hum",
                "windspeed",
                "season",
                "workingday",
                "yr",
            ],
            "categorical": ["season", "workingday", "yr"],
        },
        "Model 4: Temp x season interaction": {
            "features": [
                "temp",
                "hum",
                "windspeed",
                "season_2",
                "season_3",
                "season_4",
                "temp_season_2",
                "temp_season_3",
                "temp_season_4",
                "workingday",
                "yr",
            ],
            "categorical": ["workingday", "yr"],
        },
    }

    results = []
    for name, spec in cv_specs.items():
        df_used = (
            day_cv if name == "Model 4: Temp x season interaction" else day
        )
        mean_rmse, sd_rmse = cv_rmse(df_used, spec["features"], spec["categorical"])
        results.append(
            {"Model": name, "CV_RMSE_mean": mean_rmse, "CV_RMSE_sd": sd_rmse}
        )

    return pd.DataFrame(results)


def run_cv_log(day: pd.DataFrame, day_cv: pd.DataFrame) -> pd.DataFrame:
    """
    Run 10-fold CV for log-response Models 2–4 and return a summary table.
    """
    cv_specs_log = {
        "Log Model 2": {
            "features": ["temp", "hum", "windspeed", "season", "workingday", "yr"],
            "categorical": ["season", "workingday", "yr"],
        },
        "Log Model 3": {
            "features": [
                "temp",
                "temp_sq",
                "hum",
                "windspeed",
                "season",
                "workingday",
                "yr",
            ],
            "categorical": ["season", "workingday", "yr"],
        },
        "Log Model 4": {
            "features": [
                "temp",
                "hum",
                "windspeed",
                "season_2",
                "season_3",
                "season_4",
                "temp_season_2",
                "temp_season_3",
                "temp_season_4",
                "workingday",
                "yr",
            ],
            "categorical": ["workingday", "yr"],
        },
    }

    results = []
    for name, spec in cv_specs_log.items():
        df_used = day_cv if name == "Log Model 4" else day
        mean_rmse, sd_rmse = cv_rmse(
            df_used,
            spec["features"],
            spec["categorical"],
            response="log_cnt",
        )
        results.append(
            {
                "Model": name,
                "CV_RMSE_mean_log_scale": mean_rmse,
                "CV_RMSE_sd_log_scale": sd_rmse,
            }
        )

    return pd.DataFrame(results)
