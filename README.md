# 🌦️ Colombo Weather: Ensemble Learning Dashboard

**Module:** Machine Learning and Pattern Recognition, Hands-on Project
**Goal:** Predict tomorrow's maximum temperature in Colombo, Sri Lanka, using ensemble learning (Bagging, Boosting, Stacking and a Hybrid model), and present the results in an interactive Streamlit dashboard.

- 🔗 Live dashboard: https://224033gmlassignment1-ofuf76zqb7ckfdulamtjbs.streamlit.app/
- 📓 Notebook: `224033G_assignment1.ipynb`

## Dataset
Daily weather for Colombo (6.93°N, 79.86°E) from the free [Open-Meteo Historical Weather API](https://open-meteo.com/), 2015 to 2025 (3,949 days after feature engineering).
Variables: max/min temperature, rainfall, max wind speed.
The raw data is in `colombo_weather.csv`.

## Features
Today's max/min temperature, rainfall and wind, max temperature 1/2/3/7 days ago, 7-day rolling average temperature and rainfall, and seasonality (month, day-of-year sine/cosine). All features use only past or present data, so there is no data leakage.

## Methods
| Approach | Models |
|---|---|
| Bagging | Bagging (decision trees), Random Forest |
| Boosting | Gradient Boosting, XGBoost, LightGBM, CatBoost |
| Voting | Average of RF + LightGBM + CatBoost |
| Stacking | RF + LightGBM + CatBoost, with a Ridge meta-learner |
| Hybrid | ARIMA (time pattern) + Random Forest on ARIMA's residuals |
| Baselines | Persistence (tomorrow = today), Linear Regression, single Decision Tree, ARIMA alone |

**Evaluation:** chronological 80/20 train/test split (train on the past, test on the future) and 5-fold time-series cross-validation. Metrics: RMSE, MAE and R².

## Key results (test set, RMSE in °C, lower is better)
| Model | RMSE |
|---|---|
| Stacking (Ridge meta) | 0.947 |
| Voting | 0.949 |
| Hybrid ARIMA + RF | 0.949 |
| CatBoost | 0.950 |
| LightGBM | 0.965 |
| Random Forest | 0.965 |
| Bagging (trees) | 0.966 |
| Gradient Boosting | 0.969 |
| XGBoost | 0.970 |
| Linear Regression | 0.978 |
| ARIMA only | 0.995 |
| Persistence | 1.103 |
| Decision Tree | 1.295 |

**Time-series cross-validation (5 folds, RMSE)**
| Model | Mean | Std |
|---|---|---|
| CatBoost | 0.851 | 0.064 |
| Linear Regression | 0.857 | 0.075 |
| Random Forest | 0.863 | 0.065 |
| LightGBM | 0.898 | 0.056 |

**Findings**
- Bagging cuts the variance of a single overfitting tree (1.295 to about 0.966).
- The hybrid improves on ARIMA alone (0.995 to 0.949) because Random Forest uses rainfall and wind, which ARIMA cannot.
- The top four models (Stacking, Voting, Hybrid, CatBoost) are within 0.003 °C of each other, so they are effectively tied.
- Under time-series cross-validation, CatBoost, Linear Regression and Random Forest are statistically close. Colombo's temperature is stable and seasonal, so ensembles give modest gains.
- The live prediction page uses CatBoost: nearly as accurate as Stacking, and much faster.

## Run it locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project files
| File | Purpose |
|---|---|
| `224033G_assignment1.ipynb` | Data collection, features, modelling, evaluation |
| `app.py` | Streamlit dashboard |
| `best_model.joblib` | Trained CatBoost model used for live predictions |
| `colombo_weather.csv` | Raw data from Open-Meteo |
| `features.csv`, `results.csv`, `cv_results.csv`, `predictions.csv`, `importance.csv` | Outputs used by the dashboard |
| `requirements.txt` | Python dependencies |
