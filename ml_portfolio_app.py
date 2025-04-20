import streamlit as st
import yfinance as yf
import pandas as pd
import datetime

# Judul Aplikasi
st.title("Analisis Data BTC, Emas, dan SPY dari Yahoo Finance")

# Rentang tanggal
start_date = st.date_input("Tanggal Mulai", datetime.date(2020, 1, 1))
end_date = st.date_input("Tanggal Akhir", datetime.date.today())

# Simbol dari Yahoo Finance
symbols = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Gold (Emas - GC=F)": "GC=F",
    "SPY (S&P 500 ETF)": "SPY"
}

# Pilihan aset
selected_assets = st.multiselect(
    "Pilih aset yang ingin ditampilkan:",
    list(symbols.keys()),
    default=list(symbols.keys())
)

# Ambil data
@st.cache_data
def load_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    data = data[['Adj Close']]
    data.columns = [symbol]
    return data

# Gabungkan data
if selected_assets:
    df_list = [load_data(symbols[asset], start_date, end_date) for asset in selected_assets]
    merged_data = pd.concat(df_list, axis=1)
    st.subheader("Data Harga Penutupan Disesuaikan (Adj Close)")
    st.dataframe(merged_data.tail())

    st.subheader("Visualisasi Harga")
    st.line_chart(merged_data)

else:
    st.warning("Pilih setidaknya satu aset untuk ditampilkan.")
