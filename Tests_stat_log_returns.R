# Importe les données depuis un fichier CSV
df_log_returns <- read.csv("./Df_log_returns", header = TRUE, stringsAsFactors = FALSE)

# On supprime la colonne Date pour les analyses
df_log_returns <- df_log_returns[, -1]

#On remplace les NA par des 0, VERIFIER POURQUOI IL Y A DES NA !
df_log_returns <- replace(df_log_returns, is.na(df_log_returns), 0)

#Test de normalité Shapiro-Wilk et Jarque-Bera
library(tseries)

normality_test_results <- apply(df_log_returns, 2, function(x) {
  list(
    Shapiro_Wilk = shapiro.test(x),
    Jarque_Bera = jarque.bera.test(x)
  )
})

# On affiche les p-values des tests
sapply(normality_test_results, function(x) c(Shapiro_Wilk = x$Shapiro_Wilk$p.value, Jarque_Bera = x$Jarque_Bera$p.value))

#Test d'homoscedasticité
library(lmtest)

# Appliquer le test de Breusch-Pagan
#homoscedasticity_test_results <- apply(df_log_returns, 2, function(x) {
#  model <- lm(x ~ 1)
#  bptest(model)})

# Afficher les p-values
#sapply(homoscedasticity_test_results, function(x) x$p.value)




