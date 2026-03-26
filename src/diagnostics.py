"""
diagnostics.py
Residual analysis, VIF, influence diagnostics, robust SE, and bootstrap
utilities for bike demand modeling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson


# Statistical diagnostic summary

def diagnostic_summary(model, model_name: str) -> pd.DataFrame:
    """
    Return a one-row DataFrame with key diagnostic statistics for a fitted
    OLS model: Breusch-Pagan p-value, Jarque-Bera p-value, Durbin-Watson
    statistic, residual skewness/kurtosis, and condition number.
    """
    resid = model.resid
    exog = model.model.exog

    bp_lm, bp_lm_pvalue, bp_fvalue, bp_f_pvalue = het_breuschpagan(resid, exog)
    jb = stats.jarque_bera(resid)

    return pd.DataFrame(
        {
            "Model": [model_name],
            "Breusch_Pagan_p": [bp_f_pvalue],
            "Jarque_Bera_p": [jb.pvalue],
            "Durbin_Watson": [durbin_watson(resid)],
            "Skewness": [stats.skew(resid)],
            "Kurtosis": [stats.kurtosis(resid, fisher=False)],
            "Condition_Number": [model.condition_number],
        }
    )


# Diagnostic plots

def diagnostic_plots(model, model_name: str, x_for_resid=None, x_label: str = None):
    """
    Produce a 2x2 panel of standard regression diagnostics:
    1. Residuals vs Fitted
    2. Normal Q-Q
    3. Scale-Location
    4. Residuals vs a selected predictor (optional)
    """
    fitted = model.fittedvalues
    resid = model.resid
    std_resid = model.get_influence().resid_studentized_internal
    sqrt_abs_std_resid = np.sqrt(np.abs(std_resid))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # 1. Residuals vs Fitted
    sns.scatterplot(x=fitted, y=resid, alpha=0.7, ax=axes[0, 0])
    axes[0, 0].axhline(0, color="red", linestyle="--")
    axes[0, 0].set_title(f"{model_name}: Residuals vs Fitted")
    axes[0, 0].set_xlabel("Fitted values")
    axes[0, 0].set_ylabel("Residuals")

    # 2. Normal Q-Q
    stats.probplot(std_resid, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title(f"{model_name}: Normal Q-Q")

    # 3. Scale-Location
    sns.scatterplot(x=fitted, y=sqrt_abs_std_resid, alpha=0.7, ax=axes[1, 0])
    axes[1, 0].set_title(f"{model_name}: Scale-Location")
    axes[1, 0].set_xlabel("Fitted values")
    axes[1, 0].set_ylabel("Sqrt(|Std residuals|)")

    # 4. Residuals vs selected predictor
    if x_for_resid is not None:
        sns.scatterplot(x=x_for_resid, y=resid, alpha=0.7, ax=axes[1, 1])
        axes[1, 1].axhline(0, color="red", linestyle="--")
        axes[1, 1].set_title(f"{model_name}: Residuals vs {x_label}")
        axes[1, 1].set_xlabel(x_label)
        axes[1, 1].set_ylabel("Residuals")
    else:
        axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()


def dw_summary(model, model_name: str):
    """Print the Durbin-Watson statistic for a model."""
    print(f"{model_name} Durbin-Watson: {durbin_watson(model.resid):.3f}")


# Multicollinearity (VIF)

def compute_vif(day: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    """
    Compute Variance Inflation Factors for the specified columns.
    Defaults to the four continuous predictors.
    """
    if cols is None:
        cols = ["temp", "atemp", "hum", "windspeed"]

    vif_df = day[cols].copy()
    vif_df = sm.add_constant(vif_df)

    return pd.DataFrame(
        {
            "variable": vif_df.columns,
            "VIF": [
                variance_inflation_factor(vif_df.values, i)
                for i in range(vif_df.shape[1])
            ],
        }
    )


# Influence diagnostics

def influence_table(model, model_name: str, n_obs: int, top_n: int = 10) -> pd.DataFrame:
    """
    Return the top-n observations by Cook's distance for a fitted model.
    """
    influence = model.get_influence()
    df = pd.DataFrame(
        {
            "obs": np.arange(n_obs),
            "cooks_d": influence.cooks_distance[0],
            "leverage": influence.hat_matrix_diag,
            "std_resid": influence.resid_studentized_internal,
        }
    )
    df["model"] = model_name
    return df.sort_values("cooks_d", ascending=False).head(top_n)


def cooks_plot(model, model_name: str):
    """Stem plot of Cook's distances for all observations."""
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]

    plt.figure(figsize=(7, 4.5))
    plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",")
    plt.title(f"Cook's Distance: {model_name}")
    plt.xlabel("Observation")
    plt.ylabel("Cook's distance")
    plt.tight_layout()
    plt.show()


# Robust standard errors

def robust_se_table(model, model_name: str, cov_type: str = "HC3") -> pd.DataFrame:
    """
    Compare classical and HC3-robust standard errors and p-values.
    """
    robust = model.get_robustcov_results(cov_type=cov_type)
    return pd.DataFrame(
        {
            "model": model_name,
            "term": model.params.index,
            "coef": model.params.values,
            "SE_classical": model.bse.values,
            f"SE_{cov_type}": robust.bse,
            "p_classical": model.pvalues.values,
            f"p_{cov_type}": robust.pvalues,
        }
    )


# Bootstrap standard errors

def bootstrap_se(
    day: pd.DataFrame,
    formula: str,
    B: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Estimate bootstrap standard errors for a given OLS formula.

    Returns a DataFrame with columns: term, coef, SE_bootstrap.
    """
    rng = np.random.default_rng(seed)
    ref_model = smf.ols(formula, data=day).fit()
    coef_names = ref_model.params.index.tolist()
    boot_coefs = np.zeros((B, len(coef_names)))

    for b in range(B):
        sample_idx = rng.choice(len(day), size=len(day), replace=True)
        boot_sample = day.iloc[sample_idx].copy()
        boot_model = smf.ols(formula, data=boot_sample).fit()
        boot_coefs[b, :] = boot_model.params.reindex(coef_names).values

    return pd.DataFrame(
        {
            "term": coef_names,
            "coef": ref_model.params.values,
            "SE_bootstrap": boot_coefs.std(axis=0, ddof=1),
        }
    )
