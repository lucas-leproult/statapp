import yfinance as yf
import pandas as pd

# Définition des tickers et des noms des entreprises pour la sélection initiale
tickers_titres = {
    "AC.PA": "Accor",
    "AI.PA": "Airbus",
    "AIR.PA": "Air Liquide",
    "ORA.PA": "Orange",
    "ENGI.PA": "Engie",
    "SGO.PA": "Saint-Gobain",
    "VIE.PA": "Veolia",
    "BNP.PA": "BNP Paribas",
    "CAP.PA": "Capgemini",
    "SAF.PA": "Safran",
    "CA.PA": "Crédit Agricole",
    "BN.PA": "Danone",
    "MC.PA": "LVMH",
    "SAN.PA": "Sanofi",
    "SU.PA": "Schneider Electric",
    "TTE.PA": "TotalEnergies",
    "VIV.PA": "Vivendi",
    "^TYX": "US 30 Year T-Bond",
    "CAC.PA": "Amundi CAC 40 ETF"  # Benchmark
}

tickers = list(tickers_titres.keys())

# Définition de la période d'analyse
start_date = "2014-01-01"
end_date = "2016-12-31"

# Téléchargement des données de marché via Yahoo Finance
df_all = yf.download(tickers, start=start_date, end=end_date)["Close"].dropna()

# Séparation des données du benchmark et des actifs du portefeuille
df_benchmark = df_all["CAC.PA"]
df_portfolio_1 = df_all.drop(columns=["CAC.PA"])

# Calcul des rendements journaliers
returns_benchmark = df_benchmark.pct_change().dropna()
returns_portfolio_1 = df_portfolio_1.pct_change().dropna()

# Alignement des dates
returns_benchmark = returns_benchmark.loc[returns_portfolio_1.index]

#Création des fichiers CSV
df_portfolio_1.to_csv("Df_portfolio_1", index=False)
df_benchmark.to_csv("Df_benchmark", index=False)
returns_portfolio_1.to_csv("returns_portfolio_1", index=False)
returns_benchmark.to_csv("returns_benchmark", index=False)

