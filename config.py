"""
Configuration et constantes pour la plateforme d'analyse financière
"""

# Constantes mathématiques
TRADING_DAYS_PER_YEAR = 252  # Nombre de jours de trading par an
HOURS_PER_DAY = 24
MINUTES_PER_HOUR = 60

# Paramètres par défaut des indicateurs techniques
DEFAULT_SMA_SHORT = 20
DEFAULT_SMA_LONG = 50
DEFAULT_EMA_FAST = 12
DEFAULT_EMA_SLOW = 26
DEFAULT_EMA_SIGNAL = 9
DEFAULT_RSI_PERIOD = 14
DEFAULT_BOLLINGER_PERIOD = 20
DEFAULT_BOLLINGER_STD = 2
DEFAULT_ATR_PERIOD = 14

# Paramètres de backtesting
DEFAULT_INITIAL_CAPITAL = 1000
DEFAULT_TRANSACTION_FEE = 0.001  # 0.1%
RISK_FREE_RATE = 0.0  # Taux sans risque (simplifié)

# Actifs financiers prédéfinis
PREDEFINED_TICKERS = {
    "Actions US": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"],
    "Indices": ["^GSPC", "^DJI", "^IXIC", "^FTSE", "^GDAXI"],
    "Cryptomonnaies": ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD"],
    "Forex": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"]
}

# Intervalles de temps disponibles
TIME_INTERVALS = {
    "1 minute": "1m",
    "5 minutes": "5m",
    "15 minutes": "15m",
    "1 heure": "1h",
    "1 jour": "1d",
    "1 semaine": "1wk",
    "1 mois": "1mo"
}

# Limitations des intervalles (Yahoo Finance)
INTERVAL_LIMITATIONS = {
    "1m": {"max_days": 7, "description": "1 minute (max 7 jours)"},
    "5m": {"max_days": 60, "description": "5 minutes (max 60 jours)"},
    "15m": {"max_days": 60, "description": "15 minutes (max 60 jours)"},
    "1h": {"max_days": 730, "description": "1 heure (max 2 ans)"},
    "1d": {"max_days": 36500, "description": "1 jour (données historiques complètes)"},
    "1wk": {"max_days": 36500, "description": "1 semaine (données historiques complètes)"},
    "1mo": {"max_days": 36500, "description": "1 mois (données historiques complètes)"}
}

# Configuration des couleurs (thème professionnel)
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "danger": "#d62728",
    "warning": "#ff9800",
    "info": "#17a2b8",
    "buy_signal": "#00ff00",
    "sell_signal": "#ff0000",
    "background_dark": "#0e1117",
    "background_light": "#ffffff"
}

# Configuration Streamlit
PAGE_TITLE = "📈 Plateforme d'Analyse Financière"
PAGE_ICON = "📈"
LAYOUT = "wide"

# Messages d'aide
HELP_MESSAGES = {
    "rendements_arithmetiques": "Rendements arithmétiques : R_t = (P_t - P_{t-1}) / P_{t-1}",
    "rendements_log": "Rendements logarithmiques : r_t = ln(P_t / P_{t-1})",
    "volatilite": "Volatilité annualisée : σ_annual = σ_daily × √252",
    "sharpe": "Ratio de Sharpe : (E[R] - R_f) / σ × √252",
    "drawdown": "Maximum Drawdown : Perte maximale depuis le pic historique",
    "rsi": "RSI > 70 = Sur-acheté, RSI < 30 = Sur-vendu",
    "bollinger": "Prix touche bande supérieure = sur-acheté, bande inférieure = sur-vendu"
}
