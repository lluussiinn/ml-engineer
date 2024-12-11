from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# LightGBM hyperparameters
lgb_params = {
    'regressor': [lgb.LGBMRegressor(random_state=42)],
    'regressor__n_estimators': [100, 200, 300],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__num_leaves': [31, 50, 70],
    'regressor__max_depth': [-1, 5, 10],
}

# XGBoost hyperparameters
xgb_params = {
    'regressor': [xgb.XGBRegressor(random_state=42)],
    'regressor__n_estimators': [100,200, 300],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__max_depth': [3, 6, 10],
    'regressor__subsample': [0.7, 1.0]
}

# Random Forest hyperparameters
rf_params = {
    'regressor': [RandomForestRegressor(random_state=42)],
    'regressor__n_estimators': [100, 300],
    'regressor__max_depth': [5, 10, 15],
    'regressor__min_samples_split': [2, 5],
    'regressor__min_samples_leaf': [1, 2],
}

# XGBoost hyperparameters updated for overfitting
xgb_params_updated = {
    'regressor': [xgb.XGBRegressor(random_state=42)],
    'regressor__n_estimators': [100,200],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__max_depth': [3, 6],
    'regressor__subsample': [0.7, 1.0],
    'regressor__reg_alpha': [0.1, 1.0],
    'regressor__reg_lambda': [0.1, 1.0],
    'regressor__min_data_in_leaf': [20, 50],
    'regressor__early_stopping_rounds': [15]
}

# Initialize GridSearchCV
def perform_grid_search(model_params, X_train, y_train):
    grid_search = GridSearchCV(Pipeline(steps=[
        ('regressor', model_params['regressor'][0])]), param_grid=model_params, cv=5, scoring='neg_mean_absolute_percentage_error', verbose=1,n_jobs=-1, return_train_score=True)

    grid_search.fit(X_train, y_train)

    return grid_search
