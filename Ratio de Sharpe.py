from random import uniform

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from tabulate import tabulate
from statsmodels.tsa.stattools import *

def pprint(df):
    print(tabulate(df, headers='keys', tablefmt='psql'))



#Création d'un dictionnaire avec en clé les Tickers et en valeur le nom de l'entreprise
tickers_titres={
    "AC.PA": "Air Liquide",
    "MC.PA": "LVMH",
    "SAN.PA": "Sanofi",
    "CS.PA": "Axa",
    "OR.PA": "L'Oréal",
    "EL.PA": "Essilor Luxotica",
    #"TTE.PA": "Total Energies",
    #"SU.PA": "Schneider Electric",
    "VIE.PA": "Veolia Environnement", #modifie considérablement les poids
    "ORA.PA": "Orange", #modifie considérablement les poids
    "VIV.PA": "Vivendi",
    "SAF.PA": "Safran",
    
    "^TYX": "US 30 Year T-Bond"
}


#Récupérations de toutes les données sur la période
tickers=tuple(tickers_titres.keys())
df_all = yf.download(tickers, start="2014-01-01", end="2016-12-31")

#Extractions des cours de clôture journaliers
df_close = df_all["Close"]

#Calcul des rendements
df_returns=(df_close-df_close.shift(1))/df_close.shift(1)
df_returns=df_returns[1:].reset_index(drop=True)

df_close_returns = pd.concat([df_close, df_returns], axis=1, keys=["Close", "Returns"])


dictStats = {}
for col in df_returns.columns:
    dictStats[col] = {
        'Mean' : float(df_returns[col].mean()),
        'Skewness': float(df_returns[col].skew()),
        'Kurtosis': float(df_returns[col].kurtosis()),
    }

#Visualisation des rendements

#plt.figure(figsize=(10, 6))
#plt.plot(df_returns["SAN.PA"], label="Rendements de {0}".format(tickers_titres["SAN.PA"]), color="blue")
#plt.title("Rendements de {0}".format(tickers_titres["SAN.PA"]))
#plt.xlabel("Date")
#plt.ylabel("Rendements")
#plt.legend()
#plt.show()

################ Annualisation ################
n_days = 252  # Moyenne des jours ouvrés sur les marchés financiers

# 1. Rendements moyens annualisés
mean_returns = df_returns.mean() * n_days

# 2. Matrice de covariance annualisée
cov_matrix = df_returns.cov() * n_days



################ Optimisation numerique avec minimize de scipy.optimize  ################

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    portfolio_return = np.dot(weights, mean_returns)  # Rendement du portefeuille
    portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)  # Volatilité du portefeuille
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio

# Paramètres de l'optimisation
num_assets = len(mean_returns)
risk_free_rate = 0.02
initial_weights = np.ones(num_assets) / num_assets  # Poids égaux comme point de départ

# Contraintes : la somme des poids doit être égale à 1
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

# Bornes : chaque poids doit être compris entre 1% et 40%
bounds = tuple((0.01, 0.4) for _ in range(num_assets))

# Minimiser le ratio de Sharpe négatif
optimized_result = minimize(
    negative_sharpe_ratio,
    initial_weights,
    args=(mean_returns, cov_matrix, risk_free_rate),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)



# Résultat de l'optimisation
optimal_weights = optimized_result.x
# Calculer le ratio de Sharpe optimal
optimal_sharpe_ratio = -optimized_result.fun

print(optimal_sharpe_ratio)


tickersWeights = list(zip(tickers, optimal_weights))
for ticker, weight in tickersWeights:
        print(ticker, weight)


# Ecarts_types
std_devs = np.sqrt(np.diagonal(cov_matrix))

correlation_matrix = cov_matrix / (std_devs[:, None] * std_devs)
print(correlation_matrix)

def testKPSS(ticker):
    # Perform the KPSS test
    df = df_returns[ticker].dropna()
    kpss_stat, p_value, lags, critical_values = kpss(df, regression='c')

    # Print results
    print("KPSS Statistic:", kpss_stat)
    print("P-value:", p_value)
    print("Critical Values:", critical_values)

    # Interpret the results
    if p_value < 0.05:
        print("The series is likely non-stationary (reject the null hypothesis).")
    else:
        print("The series is likely stationary (fail to reject the null hypothesis).")

def testADF(ticker):
    df = df_returns[ticker].dropna()
    result = adfuller(df)

    print("ADF Statistic:", result[0])
    print("P-value:", result[1])
    print("Critical Values:", result[4])

    if result[1] < 0.05:
        print("The series is likely stationary (reject the null hypothesis).")
    else:
        print("The series is likely non-stationary (fail to reject the null hypothesis).")

def testHomoscedascity():
    for ticker in tickers_titres.keys():
        testKPSS(ticker)
        testADF(ticker)
        print('\n')

testHomoscedascity()