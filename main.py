import pandas as pd
import numpy as np
import yfinance as yf
from data_loader import download_data
from metrics import all_metrics_together
from features_engeneering import datasets_split, fine_tuning
from sklearn.metrics import accuracy_score, classification_report
from backtest import backtest


start_date = "2008-01-01"
end_date = "2026-01-01"
stock = input("Type your stock:").strip().upper()

# DOWNLOAD PRICES AND VOLUME
dataset, prices = download_data(stock, start_date, end_date)

# NEW DATASET WITH ALL THE METRICS AND INDICATORS
dataset = all_metrics_together(dataset, prices)

#=====================================
# SPLIT THE DATASETS
#=====================================
len_initial_train, x_train, y_train, x_test, y_test = datasets_split(dataset)

x = dataset.drop(columns = "Target")
y = dataset["Target"]

#=====================================
# FINE TUNING USING GRIDSEARCH
#=====================================
models, model_names = fine_tuning()

#=====================================
# PRINTING INITIAL MODEL STATS
#=====================================
print("MODEL'S DATA BEFORE BACKTEST")
scores = []
for i in range(len(models)):
    models[i].fit(x_train, y_train)
    score = models[i].best_score_
    scores.append(score)
    prediction = models[i].predict(x_test)
    accuracy = accuracy_score(y_test, prediction)
    report = classification_report(y_test, prediction)
    print(f"Best Parameters Found for {model_names[i]}: {models[i].best_params_}")
    print(f"{model_names[i]}'s Accuracy: {accuracy:.4f}")
    print(f"{model_names[i]}'s Classification Report: {report}")
    print("="*60)

  # Features Importance
best_xgb = models[0].best_estimator_

features = x.columns.tolist()
importances_gb = best_xgb.feature_importances_

df_gb = pd.DataFrame({"Metrics": features, 
                    "Weight": importances_gb}).sort_values(by = "Weight", ascending = False)


best_lr = models[1].best_estimator_

print(f"FEATURE IMPORTANCES {model_names[0]}")
print(df_gb)
print("-"*50)
baseline = y_test.mean()
print(f"Probability of the stock going up: {baseline:.2f}")



print("="* 50)
print("PARTE DI TRAINING INZIALE COMPLETATA")
print("="*50)
print("INIZIO BACKTEST...")

backtest(dataset, best_xgb, best_lr, len_initial_train, model_names)