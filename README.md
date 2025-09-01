# Integrating Multi-Frequency Trading Signals for Optimal Market Timing

## Overview
This project explores a **machine learning approach** to short-term stock prediction:  
> **Task**: Predict whether the next day’s closing price will be higher than the opening price.  

By integrating **high-frequency intraday equity data** and **daily options data**, the project compares multiple models — from tree ensembles to deep learning architectures — for classification performance. The study highlights how multi-frequency signals can improve market timing strategies.

---

## Dataset
- **Equity Prices (Bloomberg)**  
  - Stocks: NVDA, AMD, ARM, SMCI, SONO, SOXX, QQQ, SPY, TSLA, TSM  
  - Frequency: 8-minute OHLCV bars (Open, High, Low, Close, Volume)

- **Options Data (Barchart)**  
  - Daily features: implied volatility, put/call volume, option volume, open interest  

**Period**: Oct 10, 2023 – Apr 26, 2024:contentReference[oaicite:1]{index=1}  

---

## Preprocessing
- **Missing values**: Linear interpolation (equities), forward fill (options)  
- **Outliers**: Removed using Z-score (>3σ)  
- **Normalization**: Min-max scaling applied to all numeric features  
- **Feature engineering**:  
  - Technical indicators (MA, RSI, Bollinger Bands)  
  - Put-call ratios & option sentiment metrics  
  - Normalized prices vs moving averages  
  - Volume-weighted indicators  

---

## Methodology
Models trained on engineered features to classify next-day movement (`y=1` if Close > Open, else `y=0`):  
1. **XGBoost** – Ensemble boosting trees for structured tabular features  
2. **Random Forest** – Bagged decision trees with majority voting  
3. **LSTM** – Sequential model capturing temporal dependencies in 8-min bars  
4. **Feedforward Neural Network (FFN)** – Dense layers on engineered factors  
5. **CNN** – Convolutions on time-ordered intraday signals  

**Evaluation metrics**: Accuracy, Precision, Recall, F1, ROC-AUC:contentReference[oaicite:2]{index=2}  

---

## Key Takeaways
- **Integrating intraday + options signals** enriches predictive power beyond single-source data.  
- **LSTM and CNN** models capture sequential dependencies but require careful tuning.  
- **XGBoost** provides robust baseline performance and interpretability.  
- Feature engineering (normalized spreads, sentiment proxies) is as important as model choice.  

---

## Files
- `Integrating Multi-Frequency Trading Signals for Optimal Market Timing A Machine Learning Approach.ipynb` – Full implementation (data preprocessing, feature engineering, model training & evaluation).  
- `Integrating Multi-Frequency Trading Signals for Optimal Market Timing A Machine Learning Approach.pdf` – Project report summary.  

---

## Requirements
- Python 3.x  
- `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `xgboost`, `tensorflow/keras`

