#!/usr/bin/env python
# coding: utf-8

#L'objectif de ce code est de récuperer les cours des 18 actions du portefeuille et du T-bond

#Avant de commencer, installez :
#yfinance : py -m pip install yfinance
#matplotlib : py -m pip install matplotlib


import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#definir ticker
#Création d'une bibliothèque avec en clé les Tickers et en valeur le nom de l'entreprise
tickers_titres={
    "AC.PA":"Accor",
    "AI.PA":"Air Liquide",
    "AIR.PA":"Airbus",
    "AC.PA":"Accor",
    "ORA.PA":"Orange",
    "ENGI.PA":"Engie",
    "SGO.PA":"Saint-Gobain",
    "CS.PA":"Axa",
    "BNP.PA":"BNP Paribas",
    "DG.PA":"Vinci",
    "PUB.PA":"Publicis Groupe SA",
    "CA.PA":"Carrefour",
    "BN.PA":"Danone",
    "MC.PA":"LVMH",
    "SAN.PA":"Sanofi",
    "SU.PA":"Schneider",
    "TTE.PA":"TotalEnergies",
    "VIV.PA":"Vivendi",
    "^TYX":"US 30 Year T-Bond"
}


#Récupérations de toutes les données sur la période
tickers=tuple(tickers_titres.keys())
df_all = yf.download(tickers, start="2014-01-01", end="2016-12-31")

#Extractions des cours de clôture journaliers
df_close = df_all["Close"]

#Calcul des rendements
df_returns=(df_close-df_close.shift(1))/df_close.shift(1)
df_returns=df_returns[1:].reset_index(drop=True)

# Calcul des rendements logarithmiques
df_log_returns = np.log(df_close / df_close.shift(1))
df_log_returns=df_log_returns[1:].reset_index(drop=True)

df_close_returns = pd.concat([df_close, df_returns, df_log_returns], axis=1, keys=["Close", "Returns", "Log returns"])

#Enregistrement des données
df_close_returns.to_csv("Df_close_returns")
df_log_returns.to_csv("Df_log_returns")

#Visualisation des rendements logarithmiques
"""
plt.figure(figsize=(10, 6))
plt.plot(log_returns["SGO.PA"], label="Rendements logarithmiques de {0}".format(tickers_titres["SGO.PA"]), color="blue")
plt.title("Rendements logarithmiques de {0}".format(tickers_titres["SGO.PA"]))
plt.xlabel("Date")
plt.ylabel("Rendement logarithmiques")
plt.legend()
plt.show()"""




####################### SMV ###############################

################ Annualisation ################
n_days = 252  # Moyenne des jours ouvrés sur les marchés financiers

# 1. Rendements moyens annualisés
mean_returns = df_log_returns.mean() * n_days

# 2. Matrice de covariance annualisée
cov_matrix = df_log_returns.cov() * n_days



################ Optimisation numerique avec minimize de scipy.optimize  ################

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    portfolio_return = np.dot(weights, mean_returns)  # Rendement du portefeuille
    portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)  # Volatilité du portefeuille
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio

# Paramètres de l'optimisation
num_assets = len(mean_returns) - 1  # Exclut le T-Bond
risk_free_rate = mean_returns["^TYX"]  # Rendement sans risque annualisé
initial_weights = np.ones(num_assets) / num_assets  # Poids égaux comme point de départ

# Contraintes : la somme des poids doit être égale à 1
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

# Bornes : chaque poids doit être compris entre 0 et 1
bounds = tuple((0, 1) for _ in range(num_assets))

# Minimiser le ratio de Sharpe négatif
optimized_result = minimize(
    negative_sharpe_ratio,
    initial_weights,
    args=(mean_returns[:-1], cov_matrix.iloc[:-1, :-1], risk_free_rate),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

# Résultat de l'optimisation
optimal_weights = optimized_result.x
# Calculer le ratio de Sharpe optimal
optimal_sharpe_ratio = -optimized_result.fun


################ DEBUT ACP  ################

data_acp = df_log_returns.iloc[:, :-1]  # Retrait du T-Bond

# Standardisation et nettoyage des données
data_acp= data_acp.dropna()   # Pour l'instant j'ai simplement supprimé les lignes avec des NaN (environ 12 lignes)
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data_acp) 

# Initialisation et execution de l'ACP
pca = PCA()
pca.fit(data_standardized)

# Les contributions des composantes principales
components = pca.transform(data_standardized)
explained_variance = pca.explained_variance_ratio_

