import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

def get_data(stocks, start, end):
    stockData = pd.DataFrame()
    for stock in stocks:
        stockData[stock] = yf.download(stock, start, end, auto_adjust = True)['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

stocklist = ['AAPL', 'MSFT', 'GOOGL', 'AMZN'] # Stocks to analyze (2 to 20 stocks)
stocks = stocklist
endDate = dt.datetime.now() - dt.timedelta(days=1)
startDate = endDate - dt.timedelta(days=300)

meanReturns, covMatrix = get_data(stocks, startDate, endDate)

weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)

# Monte Carlo Method
mc_sims = 1000 #number of simulations (100 to 10,000)
T = 365 #timeframe in days (100 to 1000)

meanM = np.full (shape=(T,len(weights)), fill_value=meanReturns)
meanM = meanM.T

portfolio_sims = np.full((T,mc_sims),fill_value=0.0)

#Initial Investment
initialPortfolio = 50000

for m in range(0,mc_sims): #MC loops
    Z = np.random.normal(size=(T,len(weights)))
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanM + np.inner(L,Z)
    portfolio_sims[:,m] = initialPortfolio * np.cumprod(np.inner(weights,dailyReturns.T)+1)

# Analysis of the simulations
checkpoint_procental_up = 2 #The checkpoint to reach (1.1 to 2.0)
checkpoint_procental_down = 0.8 #The checkpoint to avoid (0.5 to 0.95)

final_values = portfolio_sims[-1,:]
incresed = np.sum(final_values > initialPortfolio)
decresed = np.sum(final_values < initialPortfolio)
set_checkpoint_up = np.sum(final_values >= checkpoint_procental_up * initialPortfolio)
set_checkpoint_down = np.sum(final_values <= checkpoint_procental_down * initialPortfolio)
highest = np.max(final_values)
lowest = np.min(final_values)
total = len(final_values)

precent_increase = incresed / total * 100
precent_decrease = decresed / total * 100

print(f'Probability of Increase: {precent_increase:.2f}%')
print(f'Probability of Decrease: {precent_decrease:.2f}%')
print(f'Probability of reaching at least {checkpoint_procental_up}x the initial investment: {set_checkpoint_up / total * 100:.2f}%')
print(f'Probability of falling at most {checkpoint_procental_down}x the initial investment: {set_checkpoint_down / total * 100:.2f}%')
print(f'Highest final portfolio value: ${highest:,.2f}')
print(f'Lowest final portfolio value: ${lowest:,.2f}')

plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('Monte Carlo Simulation of a stock Portfolio Value over Time')
plt.show()