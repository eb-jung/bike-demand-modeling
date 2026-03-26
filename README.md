# Bike Demand Modeling

A statistical analysis of bike-sharing demand using the UCI Bike Sharing Dataset. This project builds and evaluates multiple OLS regression models to understand how weather conditions, calendar features, and their interactions drive daily rental counts.

---

## Project Structure

```
bike-demand-modeling/
├── data/
│   ├── raw/
│   │   ├── day.csv          # Daily aggregated bike-sharing data
|   |   ├── hour.csv         # Hourly aggregated bike-sharing data
│   │   └── Readme.txt       # Dataset documentation
│   └── processed/           # Reserved for cleaned/transformed outputs
├── notebooks/
│   ├── 01_eda.ipynb         # Data description and exploratory analysis
│   └── 02_modeling.ipynb    # Modeling, diagnostics, and robustness checks
├── src/
│   ├── __init__.py
│   ├── preprocess.py        # Data loading and feature engineering
│   ├── models.py            # Model fitting, comparison, and cross-validation
│   └── diagnostics.py       # Residual analysis, VIF, influence, bootstrap
├── README.md
└── requirements.txt
```

---

## Notebooks

### `01_eda.ipynb` — Data Description and EDA
Loads and preprocesses the daily bike-sharing dataset, then performs a thorough exploratory analysis. Covers the distribution of the response variable (`cnt` and `log_cnt`), time trends across 2011–2012, scatter plots of rentals against temperature, humidity, and wind speed (broken down by season), box plots across categorical predictors (season, working day, weather condition), and a correlation matrix that motivates the modeling choices.

### `02_modeling.ipynb` — Modeling, Diagnostics, and Robustness Checks
Fits four nested OLS models on both the raw count and log-transformed response, then compares them on R², AIC/BIC, Durbin-Watson, and 10-fold cross-validated RMSE. Includes full residual diagnostic plots (residuals vs. fitted, Q-Q, scale-location), a Breusch-Pagan / Jarque-Bera summary table, VIF multicollinearity check, Cook's distance influence analysis, HC3 heteroskedasticity-robust standard errors, and bootstrap SE estimation.

---

## Dataset

**Source:** [UCI Machine Learning Repository — Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)

The daily dataset (`day.csv`, 731 observations) contains:

| Feature | Description |
|---|---|
| `cnt` | Total daily bike rentals (registered + casual) |
| `temp` | Normalized temperature (Celsius, divided by 41) |
| `atemp` | Normalized feeling temperature |
| `hum` | Normalized humidity |
| `windspeed` | Normalized wind speed |
| `season` | Season (1 = Winter, 2 = Spring, 3 = Summer, 4 = Fall) |
| `yr` | Year (0 = 2011, 1 = 2012) |
| `workingday` | 1 if day is neither weekend nor holiday |
| `weathersit` | Weather category (1 = Clear … 4 = Heavy Rain/Snow) |
| `dteday` | Date |

**Engineered features:**
- `log_cnt` — log-transformed response for a more symmetric distribution
- `temp_sq` — squared temperature for quadratic curvature (Model 3)
- `temp * season` interaction terms (Model 4)

---

## Methods

Four OLS models of increasing complexity are fitted and compared:

| Model | Formula |
|---|---|
| Model 1 | `cnt ~ temp + hum + windspeed` |
| Model 2 | `cnt ~ temp + hum + windspeed + season + workingday + yr` |
| Model 3 | `cnt ~ temp + temp² + hum + windspeed + season + workingday + yr` |
| Model 4 | `cnt ~ temp × season + hum + windspeed + workingday + yr` |

All four are also estimated with `log_cnt` as the response. Model selection uses a combination of adjusted R², AIC/BIC, 10-fold CV RMSE, and diagnostic checks. Robustness is assessed with HC3 heteroskedasticity-robust standard errors and nonparametric bootstrap (B = 1000).

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scipy
statsmodels
scikit-learn
jupyter
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the notebooks in order from the project root:

```bash
# 1. Exploratory analysis
jupyter notebook notebooks/01_eda.ipynb

# 2. Modeling, diagnostics, and robustness
jupyter notebook notebooks/02_modeling.ipynb
```

Both notebooks import helper functions from `src/` via a path insert at the top of each notebook. No additional configuration is needed — just ensure the working directory is the project root or that the notebooks are launched from there.
