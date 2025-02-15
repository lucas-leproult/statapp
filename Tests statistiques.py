import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm

#Récupération des données
returns_portfolio_1 = pd.read_csv("./returns_portfolio_1")
tests=["Jarque-Bera","Shapiro-Wilk","Breusch-Pagan","KPSS","ADF"]
p_values=[]

#Test de normalité
#Jarque-Bera
p_values_JB=[]
for action in returns_portfolio_1.columns:
    stat, p_value = stats.jarque_bera(returns_portfolio_1[action])
    p_values_JB.append(p_value)
p_values.append(p_values_JB)
#Shapiro-Wilk
p_values_SW=[]
for action in returns_portfolio_1.columns:
    stat, p_value = stats.shapiro(returns_portfolio_1[action])
    p_values_SW.append(p_value)
p_values.append(p_values_SW)

#Test d'homoscédasticité
#Breusch-Pagan ?
p_values_BP=[]
for action in returns_portfolio_1.columns:
    X = sm.add_constant(returns_portfolio_1[action].shift(1).dropna())  # Regressions sur les rendements décalés
    y = returns_portfolio_1[action]
    bp_test = het_breuschpagan(y, X)
    p_values_BP.append(bp_test[1])
p_values.append(p_values_BP)

#Test de stationnarité
#KPSS
#Insérer code Samuel

#ADF
#Insérer code Samuel

#On créé un tableau avec toutes les p_valeurs
p_values_all = pd.DataFrame(p_values, columns=returns_portfolio_1.columns, index = tests)
print(p_values_all)

#Corrélation entre les titres (Heatmap)
#Insérer code Samuel
