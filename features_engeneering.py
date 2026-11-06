import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


def datasets_split(dataset):

    len_initial_train = int(len(dataset) * 0.3)
    dataset_train = dataset.iloc[:len_initial_train]

    train_size = int((len(dataset_train) * 0.8))

    x = dataset.drop(columns = "Target")
    y = dataset["Target"]
    x_train = x.iloc[:train_size].copy()
    y_train = y.iloc[:train_size]
    x_test = x.iloc[train_size :].copy()
    y_test = y.iloc[train_size :]

    # FEATURES SCALING
    # VOLUME
    vol_mean = x_train["Volume / Vol20"].mean()
    vol_std = x_train["Volume / Vol20"].std()
    x_train["Volume / Vol20"] = (x_train["Volume / Vol20"] - vol_mean) / vol_std
    x_test["Volume / Vol20"] = (x_test["Volume / Vol20"] - vol_mean) / vol_std

    # RSI
    rsi_mean = x_train["RSI"].mean()
    rsi_std = x_train["RSI"].std()
    x_train["RSI"] = (x_train["RSI"] - rsi_mean) / rsi_std
    x_test["RSI"] = (x_test["RSI"] - rsi_mean) / rsi_std

    # BOLLINGER
    pct_b_mean = x_train["PCTB"].mean()
    pct_b_std = x_train["PCTB"].std()
    x_train["PCTB"] = (x_train["PCTB"] - pct_b_mean) / pct_b_std
    x_test["PCTB"] = (x_test["PCTB"] - pct_b_mean) / pct_b_std

    return len_initial_train, x_train, y_train, x_test, y_test


def fine_tuning():
      # Engineering and fine tuning of parameters
    xgb = XGBClassifier(random_state = 42)
    lg = LogisticRegression(solver='saga', max_iter = 10000, random_state = 42)

    cvts = TimeSeriesSplit(n_splits = 5)

    xgb_grid = {
        "max_depth": [3, 4, 6, 7],
        "min_child_weight": [4, 5, 6, 7, 8],
        "n_estimators": [250, 300, 350],
        "learning_rate": [0.003 ,0.01, 0.03, 0.1, 0.3],
        "gamma": [0, 1, 2, 3, 4] 
    }

    lr_grid = {
        "l1_ratio": [0.7, 0.5, 0.3, 0.1]
    }

    search_xgb = GridSearchCV(estimator = xgb, param_grid = xgb_grid, cv = cvts, n_jobs = -1, scoring = "accuracy")
    search_lr = GridSearchCV(estimator = lg, param_grid = lr_grid, cv = cvts, n_jobs = -1, scoring = "accuracy")

    models = [search_xgb, search_lr]
    model_names = ["XGBoost", "Logistic Regression"]

    return models, model_names