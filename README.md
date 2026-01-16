# 📈 Plateforme d'Analyse Financière

**Projet de Mathématiques Appliquées à la Finance**

Une plateforme complète d'analyse financière développée avec Python et Streamlit, appliquant des concepts mathématiques rigoureux (calcul matriciel, probabilités, statistiques) à des données financières réelles.

Inspirée de **TradingView**, **Bloomberg** et **Binance**.

---

## 🎯 Fonctionnalités

### 📊 Acquisition de Données
- **Yahoo Finance** : Actions, indices, cryptomonnaies, forex
- **Binance API** : Données crypto en temps réel (optionnel)
- **Import CSV** : Support des fichiers OHLC personnalisés
- Validation automatique des données

### 📈 Analyses Mathématiques Rigoureuses

#### Calculs de Rendements
- **Rendements arithmétiques** : `R_t = (P_t - P_{t-1}) / P_{t-1}`
- **Rendements logarithmiques** : `r_t = ln(P_t / P_{t-1})`
- Représentation matricielle pour plusieurs actifs

#### Statistiques Descriptives
- Moyenne, médiane, écart-type, min, max, percentiles
- **Moments d'ordre supérieur** :
  - Skewness : `E[(R - μ)³] / σ³`
  - Kurtosis : `E[(R - μ)⁴] / σ⁴`
- **Volatilité annualisée** : `σ_annual = σ_daily × √252`

#### Tests Statistiques
- Test de normalité (Shapiro-Wilk, Jarque-Bera, Kolmogorov-Smirnov)
- QQ-plots pour évaluation visuelle
- Interprétation des p-values

#### Mesures de Risque
- **Value at Risk (VaR)** : Perte maximale attendue avec un niveau de confiance donné
- **Conditional VaR (CVaR)** : Perte moyenne au-delà de la VaR

### 📉 Indicateurs Techniques

Tous les indicateurs sont implémentés avec leurs formules mathématiques complètes :

- **SMA** (Simple Moving Average) : `SMA_n(t) = (1/n) × Σ P_{t-i}`
- **EMA** (Exponential Moving Average) : `EMA_n(t) = α × P_t + (1-α) × EMA_n(t-1)`
- **RSI** (Relative Strength Index) : Oscillateur entre 0 et 100
- **Bandes de Bollinger** : `SMA ± k × σ`
- **MACD** : `EMA_fast - EMA_slow` avec ligne de signal
- **ATR** (Average True Range) : Mesure de volatilité

### 🔬 Backtesting

Stratégie implémentée : **Croisement de moyennes mobiles (SMA Crossover)**

#### Métriques de Performance
- **Rendement total et annualisé**
- **Volatilité annualisée**
- **Ratio de Sharpe** : `(E[R] - R_f) / σ × √252`
- **Maximum Drawdown** : Perte maximale depuis le pic historique
- **Profit Factor** : Total gains / Total pertes
- **Calmar Ratio** : Rendement annualisé / |MDD|
- **Taux de réussite** des trades
- **Comparaison avec Buy & Hold**

### 📊 Visualisations Professionnelles

Graphiques interactifs avec **Plotly** :
- Chandeliers japonais (candlestick)
- Prix avec indicateurs superposés
- Volume avec code couleur
- Histogrammes de distribution
- QQ-plots
- Graphiques de backtesting avec signaux
- Drawdown
- RSI et MACD en sous-graphiques

---

## 🚀 Installation

### Prérequis
- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Étapes d'installation

1. **Cloner ou télécharger le projet**
   ```bash
   cd prjtch
   ```

2. **Créer un environnement virtuel (recommandé)**
   ```bash
   python -m venv venv
   ```

3. **Activer l'environnement virtuel**
   - Windows :
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux :
     ```bash
     source venv/bin/activate
     ```

4. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Utilisation

### Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`

### Guide d'utilisation

1. **Configurer les paramètres** (barre latérale gauche)
   - Choisir la source de données (Yahoo Finance ou CSV)
   - Sélectionner un actif financier
   - Définir la période d'analyse
   - Choisir la fréquence (1d, 1h, 5m, etc.)

2. **Charger les données**
   - Cliquer sur "🔄 Charger les données"

3. **Explorer les analyses**
   - **Graphique principal** : Visualiser le prix avec indicateurs
   - **Statistiques** : Consulter les analyses statistiques complètes
   - **Distribution** : Examiner la distribution des rendements
   - **Tests de normalité** : Vérifier si les rendements suivent une loi normale
   - **VaR** : Évaluer le risque

4. **Backtesting**
   - Configurer les paramètres de la stratégie
   - Lancer le backtest
   - Analyser les métriques de performance
   - Comparer avec Buy & Hold

---

## 📁 Structure du Projet

```
prjtch/
├── app.py                          # Application Streamlit principale
├── config.py                       # Configuration et constantes
├── requirements.txt                # Dépendances Python
├── README.md                       # Documentation (ce fichier)
├── modules/
│   ├── __init__.py
│   ├── data_loader.py             # Acquisition des données
│   ├── math_operations.py         # Calculs mathématiques
│   ├── statistics.py              # Analyses statistiques
│   ├── technical_indicators.py    # Indicateurs techniques
│   ├── backtesting.py             # Backtesting et métriques
│   └── visualizations.py          # Graphiques professionnels
└── data/
    └── example_data.csv           # Données d'exemple
```

---

## 🧮 Choix Mathématiques et Justifications

### Rendements Logarithmiques vs Arithmétiques

**Rendements arithmétiques** : `R_t = (P_t - P_{t-1}) / P_{t-1}`
- ✅ Intuitifs et faciles à interpréter
- ✅ Représentent directement le pourcentage de variation
- ❌ Non additifs dans le temps

**Rendements logarithmiques** : `r_t = ln(P_t / P_{t-1})`
- ✅ Additifs dans le temps : `r_1 + r_2 + ... + r_n = ln(P_n / P_0)`
- ✅ Symétriques (gain de 10% puis perte de 10% ≠ retour au prix initial)
- ✅ Préférés pour les modèles mathématiques
- ❌ Moins intuitifs

**Choix** : L'application permet de choisir entre les deux types selon le contexte.

### Volatilité Annualisée

Formule : `σ_annual = σ_daily × √252`

**Justification** :
- Basée sur la propriété du mouvement brownien
- La variance se scale linéairement avec le temps
- L'écart-type se scale avec la racine carrée du temps
- 252 = nombre de jours de trading par an

### Ratio de Sharpe

Formule : `Sharpe = (E[R] - R_f) / σ × √252`

**Interprétation** :
- Mesure le rendement excédentaire par unité de risque
- Sharpe > 1 : Bon rendement ajusté du risque
- Sharpe > 2 : Très bon
- Sharpe > 3 : Excellent

### Maximum Drawdown

Formule : `MDD = max[(C_t - max(C_s)) / max(C_s)]`

**Utilité** :
- Mesure la perte maximale depuis le pic historique
- Indicateur clé du risque de perte
- Utilisé pour le Calmar Ratio

---

## 🔗 Références et Inspirations

### Plateformes Professionnelles
- **TradingView** : Interface intuitive, graphiques interactifs, indicateurs techniques
- **Bloomberg Terminal** : Structure modulaire, analyses institutionnelles
- **Binance** : Dashboard de trading, données accessibles via API

### API Utilisées
- **Yahoo Finance** (`yfinance`) : Données historiques gratuites
- **Binance API** (`python-binance`) : Données crypto (optionnel)

### Librairies Python
- **Streamlit** : Framework d'application web
- **Pandas** : Manipulation de données
- **NumPy** : Calculs numériques
- **Plotly** : Visualisations interactives
- **SciPy** : Tests statistiques
- **pandas-ta** : Indicateurs techniques

---

## 📊 Exemple d'Analyse : AAPL (Apple Inc.)

### Résultats typiques sur AAPL (2022-2024)

**Statistiques des rendements** :
- Rendement moyen journalier : ~0.05%
- Volatilité annualisée : ~25%
- Skewness : Légèrement négatif (queue gauche)
- Kurtosis : Positif (queues épaisses)

**Tests de normalité** :
- Les rendements ne suivent généralement pas une distribution normale
- Présence de queues épaisses (événements extrêmes plus fréquents)

**Backtesting SMA(20/50)** :
- Rendement variable selon la période
- Généralement sous-performe Buy & Hold en marché haussier
- Peut protéger en marché baissier (sortie sur signal de vente)

---

## ⚠️ Difficultés Rencontrées et Solutions

### 1. Gestion des données manquantes
**Problème** : Certaines API retournent des données incomplètes
**Solution** : Validation systématique avec `dropna()` et fonction `validate_data()`

### 2. Calcul des frais de transaction
**Problème** : Impact significatif sur les performances
**Solution** : Implémentation de frais variables et détection des changements de position

### 3. Synchronisation des indicateurs
**Problème** : Les indicateurs ont des périodes de warm-up différentes
**Solution** : Utilisation de `shift()` pour aligner les signaux avec les rendements

### 4. Performance des graphiques
**Problème** : Lenteur avec beaucoup de données
**Solution** : Utilisation de Plotly (optimisé) au lieu de Matplotlib pour l'interactivité

---

## 🔮 Améliorations Futures

### Fonctionnalités Avancées
- [ ] Support de plusieurs actifs simultanés (portefeuille)
- [ ] Optimisation de portefeuille (Markowitz)
- [ ] Stratégies de trading supplémentaires (RSI, Bollinger, ML)
- [ ] Backtesting avec vente à découvert
- [ ] Analyse de corrélation multi-actifs
- [ ] Export des résultats en PDF

### Techniques
- [ ] Intégration de l'API Binance pour données en temps réel
- [ ] Cache des données pour améliorer les performances
- [ ] Tests unitaires pour tous les modules
- [ ] Mode sombre/clair personnalisable
- [ ] Sauvegarde des configurations utilisateur

---

## 📝 Licence

Ce projet est développé dans un cadre pédagogique pour le module **Mathématiques appliquées au traitement des données**.

---

## 👨‍💻 Auteur

Projet réalisé dans le cadre du module de Mathématiques Appliquées à la Finance.

Encadrant : **M. Hamza Saber**

---

## 🙏 Remerciements

- **Yahoo Finance** pour les données financières gratuites
- **Streamlit** pour le framework d'application
- **TradingView, Bloomberg, Binance** pour l'inspiration du design
- **M. Hamza Saber** pour l'encadrement du projet

---

## 📞 Support

Pour toute question ou problème :
1. Vérifier que toutes les dépendances sont installées
2. Consulter les messages d'erreur dans le terminal
3. Vérifier la connexion internet (pour Yahoo Finance)

---

**Bon trading et bonnes analyses ! 📈**
