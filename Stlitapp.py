import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

# 앱 제목
st.title("투자 포트폴리오 최적화 도구")

# 사용자 입력 섹션
st.sidebar.header("입력 옵션")
tickers = st.sidebar.text_input("자산 리스트 입력 (예: AAPL, MSFT, TSLA)", "AAPL, MSFT, TSLA")
start_date = st.sidebar.date_input("시작 날짜", value=pd.Timestamp("2022-01-01"))
end_date = st.sidebar.date_input("종료 날짜", value=pd.Timestamp("2023-01-01"))

if st.sidebar.button("최적화 실행"):
    tickers_list = [ticker.strip() for ticker in tickers.split(",")]

    # 주식 데이터 가져오기
    data = yf.download(tickers_list, start=start_date, end=end_date)['Adj Close']
    returns = data.pct_change().dropna()

    # 최적화 함수
    def portfolio_performance(weights, mean_returns, cov_matrix):
        returns = np.dot(weights, mean_returns)
        risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return returns, risk

    def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
        returns, risk = portfolio_performance(weights, mean_returns, cov_matrix)
        sharpe_ratio = (returns - risk_free_rate) / risk
        return -sharpe_ratio

    # 데이터 처리
    num_assets = len(tickers_list)
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    initial_weights = num_assets * [1.0 / num_assets]
    bounds = tuple((0, 1) for asset in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # 최적화 실행
    result = minimize(negative_sharpe_ratio, initial_weights,
                      args=(mean_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_weights = result.x

    # 결과 표시
    st.header("최적화 결과")
    st.write("자산별 최적 비중:")
    weights_df = pd.DataFrame({
        "자산": tickers_list,
        "최적 비중(%)": [round(weight * 100, 2) for weight in optimal_weights]
    })
    st.table(weights_df)

    # 포트폴리오 성능 계산
    portfolio_return, portfolio_risk = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
    st.write(f"예상 연간 수익률: {portfolio_return*100:.2f}%")
    st.write(f"포트폴리오 리스크(표준편차): {portfolio_risk*100:.2f}%")

    # 데이터 시각화
    st.bar_chart(weights_df.set_index("자산"))
else:
    st.write("왼쪽에서 자산 리스트와 날짜를 입력한 후 '최적화 실행' 버튼을 누르세요.")
