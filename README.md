# 📈 Stock Price Prediction System

**AI-powered time series forecasting for stocks using LSTM, Prophet, and technical indicators**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6%2B-orange)
![YFinance](https://img.shields.io/badge/YFinance-0.2%2B-lightgrey)
![GitHub last commit](https://img.shields.io/github/last-commit/somya245/stock-price-prediction)

## 🌟 Key Features
- **Multi-model Approach**: LSTM, Facebook Prophet, and ARIMA implementations
- **Technical Indicators**: RSI, MACD, Bollinger Bands integration
- **Live Data**: Real-time Yahoo Finance API integration
- **Dashboard**: Interactive Streamlit visualization
- **Performance Metrics**: RMSE, MAPE, and directional accuracy

## 🚀 Quick Start

``bash
git clone https://github.com/somya245/stock-price-prediction.git
cd stock-price-prediction
pip install -r requirements.txt

# Launch prediction dashboard
streamlit run app.py


📈 Model Performance (S&P 500)
Model	RMSE	MAPE	Direction Accuracy
LSTM	18.32	2.1%	78.4%
Prophet	22.15	2.8%	72.1%
ARIMA	25.47	3.2%	68.3%
Ensemble	16.89	1.9%	81.2%
