import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize

# 페이지 설정
st.set_page_config(page_title="최적 자산 포트폴리오 비중 계산기", layout="wide")

# 함수 정의
def fetch_data(tickers, start_date, end_date):
    """
    주어진 티커에 대한 주가 데이터를 가져옵니다.
    """
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

def calculate_annualized_return_and_volatility(weights, returns):
    """
    연간 기대 수익률과 변동성을 계산합니다.
    """
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return portfolio_return, portfolio_volatility

def calculate_sharpe_ratio(weights, returns, risk_free_rate=0.02):
    """
    샤프 비율을 계산합니다.
    """
    portfolio_return, portfolio_volatility = calculate_annualized_return_and_volatility(weights, returns)
    return (portfolio_return - risk_free_rate) / portfolio_volatility

def optimize_portfolio(returns, risk_free_rate=0.02):
    """
    최적화 알고리즘을 사용하여 포트폴리오를 최적화합니다.
    """
    num_assets = returns.shape[1]
    initial_weights = np.ones(num_assets) / num_assets
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    
    result = minimize(
        lambda weights: -calculate_sharpe_ratio(weights, returns, risk_free_rate),
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result.x

# 스트림릿 UI
st.title("최적 자산 포트폴리오 비중 계산기")
st.markdown("장기 투자 관점에서 자산의 최적 비중을 계산합니다.")

# 입력 섹션
with st.sidebar:
    st.header("입력 데이터")
    tickers = st.text_input("자산 티커(symbol)를 쉼표로 구분하여 입력하세요", "AAPL,MSFT,GOOGL,AMZN,TSLA")
    start_date = st.date_input("시작 날짜", value=pd.to_datetime("2010-01-01"))
    end_date = st.date_input("종료 날짜", value=pd.to_datetime("2023-01-01"))
    risk_free_rate = st.number_input("무위험 이자율(%)", value=2.0) / 100

# 데이터 가져오기
tickers_list = [ticker.strip() for ticker in tickers.split(',')]
if st.button("최적 포트폴리오 계산"):
    try:
        data = fetch_data(tickers_list, start_date, end_date)
        st.write("### 주가 데이터")
        st.write(data.tail())
        
        # 수익률 계산
        returns = data.pct_change().dropna()
        st.write("### 일간 수익률")
        st.write(returns.tail())
        
        # 포트폴리오 최적화
        optimal_weights = optimize_portfolio(returns, risk_free_rate)
        annual_return, annual_volatility = calculate_annualized_return_and_volatility(optimal_weights, returns)
        sharpe_ratio = calculate_sharpe_ratio(optimal_weights, returns, risk_free_rate)
        
        # 결과 출력
        st.subheader("최적 포트폴리오 비중")
        for i, ticker in enumerate(tickers_list):
            st.write(f"{ticker}: {optimal_weights[i]:.2%}")
        
        st.subheader("포트폴리오 성과")
        st.write(f"연간 기대 수익률: {annual_return:.2%}")
        st.write(f"연간 변동성: {annual_volatility:.2%}")
        st.write(f"샤프 비율: {sharpe_ratio:.2f}")
    except Exception as e:
        st.error(f"오류 발생: {e}")
