import yfinance as yf
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.stats import normaltest, norm
from statsmodels.tsa.stattools import *
from statsmodels.stats.diagnostic import *
import random
import math
from tabulate import tabulate
import matplotlib.pyplot as plt

def pprint(df):
    print(tabulate(df, headers='keys', tablefmt='psql'))

def printFull(df):
    pd.set_option('display.max_rows', len(df))
    print(df)
    pd.reset_option('display.max_rows')

def seriesMoment(tSeries):
    mean = float(tSeries.mean())
    std = float(tSeries.std())
    skew = float(tSeries.skew())
    kurt = float(tSeries.kurt()) # 3 for normal distribution
    return [mean, std, skew, kurt]

########### STATIONARY TESTS ###########

def testKPSS(tSeries, threshold = 0.05):
    stat, pVal, _, _ = kpss(tSeries)
    if pVal < threshold: #Reject the null hypothesis (H is tSeries is stationary)
        return False
    return True

def testADF(tSeries, threshold = 0.05):
    stat, pVal, _, _, _, _, _ = adfuller(tSeries)
    if pVal < threshold: #Reject the null hypothesis (H is tSeries has a unit root thus not stationary)
        return True
    return False

def testStationary(tSeries, threshold = 0.05):
    stationaryKPSS = testKPSS(tSeries, threshold = threshold)
    stationaryADF = testADF(tSeries, threshold = threshold)
    if stationaryKPSS and stationaryADF:
        print("The series is stationary.")
    elif not stationaryKPSS and not stationaryADF:
        print("The series is not stationary.")
    else:
        print("Stationary test is inconclusive")

########### NORMALITY TESTS ###########

def testAD(tSeries, threshold = 0.05):
    stat, pVal = normal_ad(tSeries)
    if pVal < threshold: #Reject the null hypothesis (H is tSeries comes from a normal distribution)
        return False
    return True

def testDAgostino(tSeries, threshold = 0.05):
    stat, pVal = normaltest(tSeries) #Reject the null hypothesis (H is tSeries comes from a normal distribution)
    if pVal < threshold:
        return False
    return True

def testNormality(tSeries, threshold = 0.05):
    normalAD = testADF(tSeries, threshold)
    normalDAgostino = testDAgostino(tSeries, threshold)
    if normalAD and normalDAgostino:
        print("The series comes from a normal distribution.")
    elif not normalAD and not normalDAgostino:
        print("The series does not come from a normal distribution.")
    else:
        print("Normality test is inconclusive")

########### DATA ###########

def getRiskFreeRate(startDate, endDate):
    n = 88 # Number of days until expiration between 91 and 85, 88 is an okay approximation
    dfRf = yf.download("^IRX", start=startDate, end=endDate)["Close"]["^IRX"].squeeze()
    riskFreeDaily = (1/(1- dfRf*n/360))**(1/n) #Comes from using the expression of P (price per $100 of par face value in discount rate formula)
    return pd.Series((riskFreeDaily/100).values)

startDate = "2014-01-01"
endDate = "2016-12-31"

riskFreeRate = getRiskFreeRate(startDate, endDate) # Value of 1% ANNUAL RATE coherent with the average over the period considered

tickersNames = {
    "AC.PA": "Air Liquide",
    "MC.PA": "LVMH",
    "SAN.PA": "Sanofi",
    "CS.PA": "Axa",
    "OR.PA": "L'Oréal",
    "EL.PA": "Essilor Luxotica",
    "TTE.PA": "Total Energies",
    "SU.PA": "Schneider Electric",
    "VIE.PA": "Veolia Environnement", #modifie considérablement les poids
    "ORA.PA": "Orange", #modifie considérablement les poids
    "VIV.PA": "Vivendi",
    "SAF.PA": "Safran",

    "TLT": "iShares 20+ Year Treasury Bond ETF"
}

def getData(tickers, startDate, endDate): #Return just the daily closing values
    tickers = tuple(tickers.keys())
    dfData = yf.download(tickers, start=startDate, end=endDate)
    dfClose = dfData["Close"]
    return dfClose

def computeReturns(dfClose, tickers):#Return the daily returns provided the data and the tickers as a dictionnary
    tickersReturns = {}

    old = dfClose.shift(1)
    dfReturns = (dfClose - old) / old
    dfReturns = dfReturns[1:].reset_index(drop=True)
    dfReturns= dfReturns.dropna()

    for ticker in tickers.keys():
        tickersReturns[ticker] = dfReturns[ticker]
    return tickersReturns

def computeMoments(tickersReturns):
    tickersMoments = {}

    for ticker, returns in tickersReturns.items():
        tickersMoments[ticker] = seriesMoment(returns)

    return tickersMoments

def portfolioReturns(tickersWeights, tickersReturns): #Given a dictionary {ticker: weight} and the daily returns computes the daily returns of the portfolio
    assert tickersWeights.keys() == tickersReturns.keys()
    returns = sum(tickersWeights[ticker] * tickersReturns[ticker] for ticker in tickersWeights)
    return returns

def uniformWeights(tickers):
    weights = {}
    size = len(tickers.keys())
    for ticker in tickers.keys():
        weights[ticker] = 1/size
    return weights

def randomAllIn(tickers):
    weights = {}
    size = len(tickers.keys())
    for ticker in tickers.keys():
        weights[ticker] = 1/size
    ticker = random.choice(list(tickers.keys()))
    weights[ticker] = 1.
    return weights


########### SHARPE RATIO ###########

def computeSharpeRatio(portfolio):
    excess = portfolio - riskFreeRate/252 #Daily risk free rate
    meanExcess = excess.mean()
    stdDev = math.sqrt(excess.var())
    return math.sqrt(252)*(meanExcess / stdDev) #Anualize the daily Sharpe Ratio

def maximizeSharpeRatio(tickersReturns):
    tickers = list(tickersReturns.keys())
    weights = []
    initWeights = uniformWeights(tickersReturns)
    for ticker in tickers:
        weights += [initWeights[ticker]]
    weights = np.array(weights)
    constraint = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0.01, 0.5) for _ in range(len(weights)))

    def objectiveFunc(weights):
        weightsDict = {ticker: weights[k] for k, ticker in enumerate(tickers)}
        ratio = computeSharpeRatio(portfolioReturns(weightsDict, tickersReturns))
        return -ratio #Return -ratio since we use the minimize function and want to maximize

    result = opt.minimize(objectiveFunc, weights, bounds=bounds, constraints=constraint)

    if result.success:
        return {tickers[i]: result.x[i] for i in range(len(weights))}
    raise ValueError("Optimization failed: " + result.message)


########### MINIMIZE PORTFOLIO VAR ###########

def computeReturnsCovar(tickersReturns):
    returnsMat = pd.DataFrame.from_dict(tickersReturns)
    returnsMean = returnsMat.mean()
    centeredReturns = returnsMat - returnsMean
    n = len(returnsMat.index) #Number of rows
    covarMat = (1/(n-1))*centeredReturns.T @ centeredReturns
    return covarMat

def getWeightsVec(weightsDict):
    return pd.DataFrame([weightsDict])

def computeWeightedVar(covarMat, weights):
    weightsVec = getWeightsVec(weights)
    var = weightsVec @ covarMat @ weightsVec.T
    return var.iloc[0, 0]

def minimizeVariance(tickersReturns):
    tickers = list(tickersReturns.keys())
    weights = []
    initWeights = randomAllIn(tickersReturns) #Uniform Weights doesn't give result
    for ticker in tickers:
        weights += [initWeights[ticker]]
    weights = np.array(weights)
    constraint = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(len(weights)))

    covMat = computeReturnsCovar(tickersReturns)

    def objectiveFunc(weights):
        weightsDict = {ticker: weights[k] for k, ticker in enumerate(tickers)}
        var = computeWeightedVar(covMat, weightsDict)
        return var

    result = opt.minimize(objectiveFunc, weights, bounds=bounds, constraints=constraint) #TODO: Use differential_evolution instead

    if result.success:
        return {tickers[i]: result.x[i] for i in range(len(weights))}
    raise ValueError("Optimization failed: " + result.message)

def maximizeVariance(tickersReturns):
    tickers = list(tickersReturns.keys())
    weights = []
    initWeights = randomAllIn(tickersReturns)
    for ticker in tickers:
        weights += [initWeights[ticker]]
    weights = np.array(weights)
    constraint = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(len(weights)))

    covMat = computeReturnsCovar(tickersReturns)

    def objectiveFunc(weights):
        weightsDict = {ticker: weights[k] for k, ticker in enumerate(tickers)}
        var = computeWeightedVar(covMat, weightsDict)
        return -var

    result = opt.minimize(objectiveFunc, weights, bounds=bounds, constraints=constraint) #TODO: Use differential_evolution instead

    if result.success:
        return {tickers[i]: result.x[i] for i in range(len(weights))}
    raise ValueError("Optimization failed: " + result.message)

data = getData(tickersNames, startDate, endDate)
returns = computeReturns(data, tickersNames)
optiWeights = maximizeSharpeRatio(returns)
portfolio = portfolioReturns(optiWeights, returns)
#print(portfolio)
#ratio = computeSharpeRatio(portfolio)
#sumVal = sum(optiWeights.values())
#print(optiWeights )
#print(ratio, computeSharpeRatio(portfolioReturns(uniformWeights(tickersNames), returns)))

#print(252*portfolioReturns(minimizeVariance(returns), returns).mean(), 252*portfolio.mean(), 252*portfolioReturns(uniformWeights(tickersNames), returns).mean())

########### SORTINO RATIO ###########

def computeSortinoRatio(portfolio):
    excess = portfolio - riskFreeRate/252 #Daily risk free rate
    negReturns = excess[excess < 0]
    meanExcess = excess.mean()
    stdDev = math.sqrt(negReturns.var())
    return math.sqrt(252)*(meanExcess / stdDev) #Anualize the daily Sharpe Ratio

def maximizeSortinoRatio(tickersReturns):
    tickers = list(tickersReturns.keys())
    weights = []
    initWeights = uniformWeights(tickersReturns)
    for ticker in tickers:
        weights += [initWeights[ticker]]
    weights = np.array(weights)
    constraint = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0.01, 0.5) for _ in range(len(weights)))

    def objectiveFunc(weights):
        weightsDict = {ticker: weights[k] for k, ticker in enumerate(tickers)}
        ratio = computeSortinoRatio(portfolioReturns(weightsDict, tickersReturns))
        return -ratio #Return -ratio since we use the minimize function and want to maximize

    result = opt.minimize(objectiveFunc, weights, bounds=bounds, constraints=constraint)

    if result.success:
        return {tickers[i]: result.x[i] for i in range(len(weights))}
    raise ValueError("Optimization failed: " + result.message)

########### VaR ###########

def computeHistVaR(portfolio, confLvl):
    VaR = -np.percentile(portfolio, 100*(1 - confLvl))
    return VaR

def computeNormalVaR(portfolio, confLvl):
    mean = portfolio.mean()
    std = portfolio.std()
    zAlpha = norm.ppf(1 - confLvl)
    print(zAlpha)
    VaR = - (mean + zAlpha*std)
    return VaR

def maximizeUnderVaR(tickersReturns, maxLoss, confLvl=0.05):
    tickers = list(tickersReturns.keys())
    weights = []
    initWeights = uniformWeights(tickersReturns)

    for ticker in tickers:
        weights.append(initWeights[ticker])

    weights = np.array(weights)

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': lambda w: maxLoss + computeHistVaR(portfolioReturns(
            {ticker: w[i] for i, ticker in enumerate(tickers)}, tickersReturns), confLvl)} #maxLoss > -VaR (computed VaR is negative so -Var is positive)
    ]

    bounds = tuple((0.01, 0.5) for _ in range(len(weights)))

    def objectiveFunc(weights):
        weightsDict = {ticker: weights[k] for k, ticker in enumerate(tickers)}
        print(computeHistVaR(portfolioReturns(weightsDict, tickersReturns), confLvl))
        print(portfolioReturns(weightsDict, tickersReturns).mean(), '\n')
        return -np.mean(portfolioReturns(weightsDict, tickersReturns))  # Negative for maximization

    result = opt.minimize(objectiveFunc, weights, bounds=bounds, constraints=constraints)

    if result.success:
        return {tickers[i]: result.x[i] for i in range(len(weights))}

    raise ValueError("Optimization failed: " + result.message)

def maximizeVaR(tickersReturns, confLvl):
    tickers = list(tickersReturns.keys())
    weights = []
    initWeights = uniformWeights(tickersReturns)
    for ticker in tickers:
        weights += [initWeights[ticker]]
    weights = np.array(weights)
    constraint = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(len(weights)))

    covMat = computeReturnsCovar(tickersReturns)

    def objectiveFunc(weights):
        weightsDict = {ticker: weights[k] for k, ticker in enumerate(tickers)}
        VaR = computeHistVaR(portfolioReturns(weightsDict, tickersReturns), confLvl)
        return -VaR #Return -VaR since we want to maximize but use the minimize function

    result = opt.minimize(objectiveFunc, weights, bounds=bounds, constraints=constraint) #TODO: Use differential_evolution instead

    if result.success:
        return {tickers[i]: result.x[i] for i in range(len(weights))}
    raise ValueError("Optimization failed: " + result.message)

def annualizeVaR(VaR):
    return math.sqrt(252)*VaR

########### CVaR ###########

def computeHistCVaR(portfolio, confLvl):
    VaR = computeHistVaR(portfolio, confLvl)
    CVaR = (portfolio[portfolio < VaR]).mean() #We don't use absolute VaR
    return CVaR

def computeNormalCVaR(portfolio, confLvl):
    mean = portfolio.mean()
    std = portfolio.std()
    zAlpha = norm.ppf(confLvl)

    CVaR = (norm.pdf(zAlpha) / confLvl) * std - mean
    return -CVaR

def maximizeUnderCVaR(tickersReturns, maxLoss, confLvl=0.05):
    tickers = list(tickersReturns.keys())
    weights = []
    initWeights = uniformWeights(tickersReturns)

    for ticker in tickers:
        weights.append(initWeights[ticker])

    weights = np.array(weights)

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': lambda w: maxLoss + computeHistCVaR(portfolioReturns(
            {ticker: w[i] for i, ticker in enumerate(tickers)}, tickersReturns), confLvl)} #maxLoss > -CVaR (computed CVaR is negative so -CVar is positive)
    ]

    bounds = tuple((0.01, 0.5) for _ in range(len(weights)))  # Ensure diversification

    def objectiveFunc(weights):
        weightsDict = {ticker: weights[k] for k, ticker in enumerate(tickers)}
        return -np.mean(portfolioReturns(weightsDict, tickersReturns))  # Negative for maximization

    result = opt.minimize(objectiveFunc, weights, bounds=bounds, constraints=constraints)

    if result.success:
        return {tickers[i]: result.x[i] for i in range(len(weights))}

    raise ValueError("Optimization failed: " + result.message)

def maximizeCVaR(tickersReturns, confLvl):
    tickers = list(tickersReturns.keys())
    weights = []
    initWeights = uniformWeights(tickersReturns)
    for ticker in tickers:
        weights += [initWeights[ticker]]
    weights = np.array(weights)
    constraint = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(len(weights)))

    covMat = computeReturnsCovar(tickersReturns)

    def objectiveFunc(weights):
        weightsDict = {ticker: weights[k] for k, ticker in enumerate(tickers)}
        CVaR = computeHistCVaR(portfolioReturns(weightsDict, tickersReturns), confLvl)
        return -CVaR #Return -VaR since we want to maximize but use the minimize function

    result = opt.minimize(objectiveFunc, weights, bounds=bounds, constraints=constraint) #TODO: Use differential_evolution instead

    if result.success:
        return {tickers[i]: result.x[i] for i in range(len(weights))}
    raise ValueError("Optimization failed: " + result.message)

#print(annualizeVaR(computeHistVaR(portfolioReturns(maximizeVaR(returns, 0.05), returns), 0.05)))

minWeights = minimizeVariance(returns)
maxWeights = maximizeVariance(returns)
covMat = computeReturnsCovar(returns)
maxRisk = math.sqrt(computeWeightedVar(covMat, maxWeights))
minRisk = math.sqrt(computeWeightedVar(covMat, minWeights))

def plotEfficientFrontier(tickersReturns, n_portfolios=5000):
    tickers = list(tickersReturns.keys())
    covMat = computeReturnsCovar(tickersReturns)

    results = np.zeros((3, n_portfolios))

    for i in range(n_portfolios):
        weights = np.random.dirichlet(np.ones(len(tickers)), size=1).flatten()
        weights_dict = {tickers[k]: weights[k] for k in range(len(tickers))}

        portfolio_return = portfolioReturns(weights_dict, tickersReturns).mean() * 252  # Annualized
        portfolio_volatility = math.sqrt(computeWeightedVar(covMat, weights_dict) * 252)  # Annualized

        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = computeSharpeRatio(portfolioReturns(weights_dict, tickersReturns))

    plt.figure(figsize=(10, 6))
    plt.scatter(results[1], results[0], c=results[2], cmap='viridis', alpha=0.5)
    plt.colorbar(label="Sharpe Ratio")
    plt.xlabel("Volatility (Standard Deviation)")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier (Monte Carlo Simulation)")
    plt.grid(True)
    plt.show()

plotEfficientFrontier(returns, 50000)
