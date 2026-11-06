import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def backtest(dataset, best_xgb, best_lr, len_initial_train, model_names):
    x_bt = dataset.drop(columns = "Target")
    y_bt = dataset["Target"]

    

    # WALK FORDWARD LOOP
    freq = 63
    models_used = [best_xgb, best_lr]

    # Lunghezza della parte di dataset disponibile per il backtest (dopo il train iniziale)
    fine_backtest = len(dataset) - len_initial_train

    market_total_returns = []
    strategy_quarterly_predictions = {name: [] for name in model_names}
    accuracies = {name: [] for name in model_names}

    for q in range(0, fine_backtest - freq, freq):
        x_train_bt = x_bt.iloc[0 : len_initial_train + q]
        y_train_bt = y_bt.iloc[0:len_initial_train + q]
        x_test_bt = x_bt.iloc[len_initial_train + q : len_initial_train + q + freq]
        y_test_bt = y_bt.iloc[len_initial_train + q : len_initial_train + q + freq]
        market_quarterly_returns = dataset["Return"].shift(-1).iloc[len_initial_train + q: len_initial_train + q + freq]
        market_total_returns.extend(market_quarterly_returns)
        

        for i in range(len((models_used))):
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train_bt)
            x_test_scaled = scaler.transform(x_test_bt) 
            models_used[i].fit(x_train_scaled, y_train_bt)

            quarterly_prediction = models_used[i].predict(x_test_scaled)
            strategy_quarterly_predictions[model_names[i]].extend(quarterly_prediction)
            accuracy = accuracy_score(y_test_bt, quarterly_prediction)
            accuracies[model_names[i]].append(accuracy)



    for name in model_names:
        df = pd.DataFrame({
            f"Predictions_{name}": strategy_quarterly_predictions[name],
            "Market_Returns": market_total_returns
        }).dropna()

        df["Strategy_Returns"] = df[f"Predictions_{name}"] * df["Market_Returns"]

        strategy_final_returns =  (1 + df["Strategy_Returns"]).cumprod()
        market_final_returns = (1 + df["Market_Returns"]).cumprod()
        mean_accuracy = np.mean(accuracies[name])

        df["Strategy_prices"] = 100 * strategy_final_returns
        strategy_final_return = (df["Strategy_prices"].iloc[-1] - df["Strategy_prices"].iloc[0]) / df["Strategy_prices"].iloc[0]

        df["Market_Prices"] = 100 * market_final_returns
        market_final_return = (df["Market_Prices"].iloc[-1] - df["Market_Prices"].iloc[0]) / df["Market_Prices"].iloc[0]

        monthly_bh_returns = []
        monthly_str_returns = []
        for p in range(23, len(df["Market_Prices"]), 23):
            monthly_market_return = (df["Market_Prices"].iloc[p]- df["Market_Prices"].iloc[p-23]) / df["Market_Prices"].iloc[p-23]
            monthly_bh_returns.append(monthly_market_return)
            monthly_str_return = (df["Strategy_prices"].iloc[p]- df["Strategy_prices"].iloc[p-23]) / df["Strategy_prices"].iloc[p-23]
            monthly_str_returns.append(monthly_str_return)


        df_win_rate = pd.DataFrame({"B&H monthly Return": monthly_bh_returns,
                                    "Strategy Monthly Return": monthly_str_returns}).dropna()

        df_win_rate["Win_Rate"] = np.where(df_win_rate["B&H monthly Return"] > df_win_rate["Strategy Monthly Return"], 0, 1)
        win_rate = df_win_rate["Win_Rate"].mean()

        # FINAL PRINTING 
        print("="*50)
        print(f"FINAL REPORT PER MODELLO: {name}")
        print("="*50)
        print(f"MODEL NAME:{name}")
        print(f"Strategy Final Return: {(strategy_final_return * 100):.2f}")
        print(f"Market Final Return: {(market_final_return * 100):.2f}")
        print(f"Win Rate(%): {win_rate:.4f}")
        print(f"Mean Accuracy: {mean_accuracy:.4f}")
        print("="*50)