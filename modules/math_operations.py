"""
Module de calculs mathématiques pour l'analyse financière
Implémentation rigoureuse des formules mathématiques
"""

import pandas as pd
import numpy as np
from typing import Union


def arithmetic_returns(prices: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """
    Calcule les rendements arithmétiques
    
    Formule mathématique:
        R_t = (P_t - P_{t-1}) / P_{t-1}
    
    Args:
        prices: Série de prix
    
    Returns:
        Série de rendements arithmétiques
    
    Justification:
        Les rendements arithmétiques sont intuitifs et faciles à interpréter.
        Ils représentent le pourcentage de variation du prix.
    """
    if isinstance(prices, pd.Series):
        return prices.pct_change().dropna()
    else:
        return np.diff(prices) / prices[:-1]


def log_returns(prices: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """
    Calcule les rendements logarithmiques
    
    Formule mathématique:
        r_t = ln(P_t / P_{t-1})
    
    Args:
        prices: Série de prix
    
    Returns:
        Série de rendements logarithmiques
    
    Justification:
        Les rendements logarithmiques sont additifs dans le temps et symétriques
        par rapport aux gains et pertes. Ils sont préférés pour les analyses
        statistiques et les modèles mathématiques.
        
        Propriété: r_1 + r_2 + ... + r_n = ln(P_n / P_0)
    """
    if isinstance(prices, pd.Series):
        return np.log(prices / prices.shift(1)).dropna()
    else:
        return np.log(prices[1:] / prices[:-1])


def returns_matrix(prices_df: pd.DataFrame, method: str = 'arithmetic') -> pd.DataFrame:
    """
    Calcule la matrice de rendements pour plusieurs actifs
    
    Args:
        prices_df: DataFrame avec les prix de plusieurs actifs (colonnes = actifs)
        method: 'arithmetic' ou 'log'
    
    Returns:
        Matrice de rendements de dimensions (p, n) où:
        - p = nombre de périodes
        - n = nombre d'actifs
    
    Représentation matricielle:
        R = [r_1, r_2, ..., r_n]
        où chaque r_i est un vecteur colonne de rendements
    """
    if method == 'arithmetic':
        return prices_df.pct_change().dropna()
    elif method == 'log':
        return np.log(prices_df / prices_df.shift(1)).dropna()
    else:
        raise ValueError("method doit être 'arithmetic' ou 'log'")


def mean_returns(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calcule la moyenne des rendements
    
    Formule mathématique:
        R̄ = (1/n) × Σ R_i
    
    Args:
        returns: Série de rendements
    
    Returns:
        Moyenne des rendements
    """
    if isinstance(returns, pd.Series):
        return returns.mean()
    else:
        return np.mean(returns)


def variance_returns(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calcule la variance des rendements
    
    Formule mathématique:
        σ² = (1/(n-1)) × Σ(R_i - R̄)²
    
    Args:
        returns: Série de rendements
    
    Returns:
        Variance des rendements
    
    Note:
        Utilise n-1 au dénominateur (variance non biaisée)
    """
    if isinstance(returns, pd.Series):
        return returns.var()
    else:
        return np.var(returns, ddof=1)


def volatility_annualized(returns: Union[pd.Series, np.ndarray], periods_per_year: int = 252) -> float:
    """
    Calcule la volatilité annualisée
    
    Formule mathématique:
        σ_annual = σ_period × √(periods_per_year)
    
    Args:
        returns: Série de rendements
        periods_per_year: Nombre de périodes par an
            - 252 pour les données journalières (jours de trading)
            - 52 pour les données hebdomadaires
            - 12 pour les données mensuelles
    
    Returns:
        Volatilité annualisée
    
    Justification:
        La volatilité se scale avec la racine carrée du temps (propriété
        du mouvement brownien). Cette formule suppose que les rendements
        sont indépendants et identiquement distribués.
    """
    if isinstance(returns, pd.Series):
        std_period = returns.std()
    else:
        std_period = np.std(returns, ddof=1)
    
    return std_period * np.sqrt(periods_per_year)


def correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la matrice de corrélation
    
    Formule mathématique:
        ρ_ij = Cov(R_i, R_j) / (σ_i × σ_j)
    
    Args:
        returns_df: DataFrame de rendements (colonnes = actifs)
    
    Returns:
        Matrice de corrélation de dimensions (n, n)
    
    Propriétés:
        - ρ_ii = 1 (corrélation d'un actif avec lui-même)
        - -1 ≤ ρ_ij ≤ 1
        - Matrice symétrique: ρ_ij = ρ_ji
    """
    return returns_df.corr()


def covariance_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la matrice de covariance
    
    Formule mathématique:
        Cov(R_i, R_j) = E[(R_i - E[R_i])(R_j - E[R_j])]
    
    Args:
        returns_df: DataFrame de rendements (colonnes = actifs)
    
    Returns:
        Matrice de covariance de dimensions (n, n)
    
    Utilisation:
        La matrice de covariance est utilisée dans la théorie moderne
        du portefeuille pour calculer le risque d'un portefeuille.
    """
    return returns_df.cov()


def cumulative_returns(returns: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """
    Calcule les rendements cumulés
    
    Formule mathématique:
        R_cum(t) = Π(1 + R_i) - 1
    
    Args:
        returns: Série de rendements
    
    Returns:
        Série de rendements cumulés
    """
    if isinstance(returns, pd.Series):
        return (1 + returns).cumprod() - 1
    else:
        return np.cumprod(1 + returns) - 1


def rolling_volatility(returns: pd.Series, window: int = 20, periods_per_year: int = 252) -> pd.Series:
    """
    Calcule la volatilité glissante (rolling volatility)
    
    Args:
        returns: Série de rendements
        window: Taille de la fenêtre glissante
        periods_per_year: Nombre de périodes par an
    
    Returns:
        Série de volatilité glissante annualisée
    """
    return returns.rolling(window).std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns: Union[pd.Series, np.ndarray], risk_free_rate: float = 0.0, 
                 periods_per_year: int = 252) -> float:
    """
    Calcule le ratio de Sharpe
    
    Formule mathématique:
        Sharpe = (E[R] - R_f) / σ × √(periods_per_year)
    
    Args:
        returns: Série de rendements
        risk_free_rate: Taux sans risque annualisé (par défaut 0)
        periods_per_year: Nombre de périodes par an
    
    Returns:
        Ratio de Sharpe annualisé
    
    Interprétation:
        - Sharpe > 1: Bon rendement ajusté du risque
        - Sharpe > 2: Très bon rendement ajusté du risque
        - Sharpe > 3: Excellent rendement ajusté du risque
    """
    if isinstance(returns, pd.Series):
        mean_return = returns.mean()
        std_return = returns.std()
    else:
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
    
    # Convertir le taux sans risque en taux périodique
    risk_free_period = risk_free_rate / periods_per_year
    
    # Calculer le ratio de Sharpe annualisé
    excess_return = mean_return - risk_free_period
    sharpe = (excess_return / std_return) * np.sqrt(periods_per_year)
    
    return sharpe
