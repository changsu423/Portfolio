import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

st.title("투자 포트폴리오 최적화 도구")

st.sidebar.header("입력 옵션")
tickers = st.sidebar.text_input("자산 리스트 입력 (예: AAPL, MSFT, TSLA)", "AAPL, MSFT, TSLA")
start_date = st.sidebar.date_input("시작 날짜", value=pd.Timestamp("2022-01-01"))
end_date = st.sidebar.date_input("종료 날짜", value=pd.Timestamp("2023-01-01"))

if st.sidebar.button("최적화 실행"):
    tickers_list = [ticker.strip() for ticker in tickers.split(",")]
    data = yf.download(tickers_list, start=start_date, end=end_date)

    if data.empty or 'Adj Close' not in data.columns:
        st.error("데이터를 가져올 수 없습니다. 티커 이름과 날짜를 다시 확인하세요.")
    else:
        adjusted_close = data['Adj Close']
        st.write(adjusted_close.head())
