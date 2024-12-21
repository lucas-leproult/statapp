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
