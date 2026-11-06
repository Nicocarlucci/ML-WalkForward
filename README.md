# 📈 FinML-WalkForward: Financial Machine Learning & Walk-Forward Backtesting

A Python-based Machine Learning framework for **Walk-Forward Backtesting** on financial time-series data. This repository features optimized feature engineering for key technical indicators (RSI, Bollinger Bands %B) to ensure realistic, leakage-free trading strategy simulations.

---

## 🎯 Project Overview

Standard cross-validation techniques often introduce **look-ahead bias** and **data leakage** when applied to financial time series. This framework implements a rolling window (**Walk-Forward Testing**) approach to train and validate machine learning models exclusively on historical data prior to each test period.

---

## 🛠️ Key Features

- **Vectorized Feature Engineering:**
  - **Relative Strength Index (RSI):** Multi-period price momentum measurement using vectorized Pandas operations.
  - **Bollinger Bands %B:** Normalized volatility indicator ($0 - 1$) tracking price location relative to upper/lower Bollinger Bands.
- **Robust Data Pipeline:**
  - Explicit single-series processing to avoid dataframe-level scale distortions.
  - Automated dynamic cleaning of window-initialization null values (`NaN`) to preserve temporal alignment.
- **Walk-Forward Engine:** Modular setup designed to retrain models dynamically across shifting time horizons.

---

## 📐 Mathematical Definitions

### 1. Relative Strength Index (RSI)
$$\text{RS} = \frac{\text{SMA}(\text{Gain}, N)}{\text{SMA}(\text{Loss}, N)}$$
$$\text{RSI} = 100 - \left( \frac{100}{1 + \text{RS}} \right)$$

### 2. Bollinger Bands %B
$$\%B = \frac{\text{Close} - \text{Lower Band}}{\text{Upper Band} - \text{Lower Band}}$$

---

## 💻 Setup & Installation

Ensure you have Python 3.9+ installed.

```bash
# Clone the repository
git clone [https://github.com/your-username/ML-WalkForward.git](https://github.com/your-username/ML-WalkForward.git)
cd FinML-WalkForward

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install pandas numpy scikit-learn xgboost