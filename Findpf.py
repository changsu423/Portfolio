import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

# 1. 자산 및 데이터 설정
tickers = ['AAPL', 'MSFT', 'GOOG']  # 원하는 자산
data = yf.download(tickers, start='2022-01-01', end='2023-01-01')['Adj Close']
returns = data.pct_change().dropna()  # 수익률 계산

# 2. 포트폴리오 성능 함수 정의
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, risk

# 3. 샤프비율 최대화를 위한 최적화
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    returns, risk = portfolio_performance(weights, mean_returns, cov_matrix)
    sharpe_ratio = (returns - risk_free_rate) / risk
    return -sharpe_ratio  # 최소화 문제로 변환

# 4. 제약조건과 초기 값
num_assets = len(tickers)
mean_returns = returns.mean()
cov_matrix = returns.cov()
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # 합이 1
bounds = tuple((0, 1) for asset in range(num_assets))  # 비중 0~1

# 최적화 실행
initial_weights = num_assets * [1.0 / num_assets]
result = minimize(negative_sharpe_ratio, initial_weights,
                  args=(mean_returns, cov_matrix),
                  method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = result.x

# 5. 결과 출력
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight*100:.2f}%")
