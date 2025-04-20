import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
import joblib
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

# --- Load Data ---
st.title("ML-Based Portfolio Allocation (BTC, Gold, SPY)")
tickers = ['BTC-USD', 'GLD', 'SPY']
data = yf.download(tickers, start="2020-01-01")['Adj Close']
data.dropna(inplace=True)
st.write("Data Harga Terbaru")
st.line_chart(data)

# --- Compute Returns ---
returns = data.pct_change().dropna()

# --- Load Trained Models ---
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("xgb_model.json")
hmm_model = joblib.load("hmm_model.pkl")

# --- Generate XGBoost Features ---
def generate_features(returns):
    df = returns.copy()
    df['ret_1d'] = returns['SPY']
    df['ret_1d_lag1'] = df['ret_1d'].shift(1)
    df['ma_5'] = df['ret_1d'].rolling(5).mean()
    df['momentum'] = df['ret_1d'] - df['ma_5']
    df['target'] = (df['ret_1d'].shift(-1) > 0).astype(int)
    return df.dropna()

feat_df = generate_features(returns)
X_feat = feat_df[['ret_1d_lag1', 'ma_5', 'momentum']]
xgb_pred = xgb_model.predict(X_feat)
xgb_series = pd.Series(xgb_pred, index=feat_df.index)

# --- HMM Regime ---
spy_ret = returns['SPY'].dropna().values.reshape(-1, 1)
regimes = hmm_model.predict(spy_ret)
regime_series = pd.Series(regimes, index=returns['SPY'].dropna().index)

# --- Align ---
common_index = regime_series.index.intersection(xgb_series.index)
regime_series = regime_series.loc[common_index]
xgb_series = xgb_series.loc[common_index]

# --- Define Allocation ---
alloc_matrix = {
    (1, 0): [0.4, 0.2, 0.4],
    (1, 1): [0.3, 0.3, 0.4],
    (1, 2): [0.2, 0.5, 0.3],
    (0, 0): [0.2, 0.5, 0.3],
    (0, 1): [0.1, 0.7, 0.2],
    (0, 2): [0.05, 0.85, 0.1]
}

weights = pd.DataFrame(index=common_index, columns=returns.columns)
for dt in common_index:
    pred = xgb_series.loc[dt]
    reg = regime_series.loc[dt]
    weights.loc[dt] = alloc_matrix.get((pred, reg), [1/3, 1/3, 1/3])
weights = weights.fillna(method='ffill')

# --- Backtest ---
aligned_returns = returns.loc[weights.index]
portfolio = (weights.shift(1) * aligned_returns).sum(axis=1)
equity_curve = (1 + portfolio).cumprod()

# --- Display ---
st.subheader("Kinerja Strategi Real-time ML")
st.line_chart(equity_curve.rename("Equity Curve"))

# --- Heatmap ---
st.subheader("Heatmap Alokasi Aset")
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(weights.astype(float), cmap="YlGnBu", ax=ax)
st.pyplot(fig)

st.caption("Model XGBoost dan HMM dilatih offline, lalu dimuat untuk prediksi harian.")
