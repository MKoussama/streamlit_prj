"""
Module d'analyses statistiques avancées
Tests de normalité, moments d'ordre supérieur, distributions
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple


def descriptive_stats(returns: pd.Series) -> Dict[str, float]:
    """
    Calcule les statistiques descriptives des rendements
    
    Args:
        returns: Série de rendements
    
    Returns:
        Dictionnaire avec les statistiques descriptives
    """
    return {
        "Moyenne": returns.mean(),
        "Médiane": returns.median(),
        "Écart-type": returns.std(),
        "Minimum": returns.min(),
        "Maximum": returns.max(),
        "Percentile 5%": returns.quantile(0.05),
        "Percentile 25%": returns.quantile(0.25),
        "Percentile 75%": returns.quantile(0.75),
        "Percentile 95%": returns.quantile(0.95),
        "Nombre d'observations": len(returns)
    }


def higher_moments(returns: pd.Series) -> Dict[str, float]:
    """
    Calcule les moments d'ordre supérieur
    
    Formules mathématiques:
        Skewness = E[(R - μ)³] / σ³
        Kurtosis = E[(R - μ)⁴] / σ⁴
    
    Args:
        returns: Série de rendements
    
    Returns:
        Dictionnaire avec Skewness et Kurtosis
    
    Interprétation:
        Skewness:
            - = 0: Distribution symétrique
            - > 0: Queue à droite (rendements positifs extrêmes)
            - < 0: Queue à gauche (rendements négatifs extrêmes)
        
        Kurtosis (excess):
            - = 0: Distribution normale
            - > 0: Queues épaisses (plus de valeurs extrêmes)
            - < 0: Queues fines (moins de valeurs extrêmes)
    """
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)  # Excess kurtosis (normal = 0)
    
    return {
        "Skewness": skewness,
        "Kurtosis": kurtosis,
        "Interprétation Skewness": interpret_skewness(skewness),
        "Interprétation Kurtosis": interpret_kurtosis(kurtosis)
    }


def interpret_skewness(skew: float) -> str:
    """Interprète la valeur de skewness"""
    if abs(skew) < 0.5:
        return "Distribution approximativement symétrique"
    elif skew > 0:
        return "Distribution asymétrique à droite (queue positive)"
    else:
        return "Distribution asymétrique à gauche (queue négative)"


def interpret_kurtosis(kurt: float) -> str:
    """Interprète la valeur de kurtosis"""
    if abs(kurt) < 0.5:
        return "Distribution proche de la normale"
    elif kurt > 0:
        return "Distribution leptokurtique (queues épaisses)"
    else:
        return "Distribution platykurtique (queues fines)"


def normality_test(returns: pd.Series, alpha: float = 0.05) -> Dict[str, any]:
    """
    Teste la normalité de la distribution des rendements
    
    Tests utilisés:
        1. Shapiro-Wilk: Bon pour les petits échantillons (n < 5000)
        2. Jarque-Bera: Basé sur Skewness et Kurtosis
    
    Hypothèses:
        H0: Les données suivent une distribution normale
        H1: Les données ne suivent pas une distribution normale
    
    Args:
        returns: Série de rendements
        alpha: Niveau de significativité (par défaut 0.05)
    
    Returns:
        Dictionnaire avec les résultats des tests
    
    Interprétation de la p-value:
        - p-value > α: On ne peut pas rejeter H0 (données possiblement normales)
        - p-value ≤ α: On rejette H0 (données non normales)
    """
    results = {}
    
    # Test de Shapiro-Wilk (si échantillon pas trop grand)
    if len(returns) < 5000:
        shapiro_stat, shapiro_pvalue = stats.shapiro(returns)
        results["Shapiro-Wilk"] = {
            "Statistique": shapiro_stat,
            "p-value": shapiro_pvalue,
            "Conclusion": "Normale" if shapiro_pvalue > alpha else "Non normale",
            "Interprétation": f"p-value = {shapiro_pvalue:.4f} {'>' if shapiro_pvalue > alpha else '≤'} {alpha}"
        }
    
    # Test de Jarque-Bera
    jb_stat, jb_pvalue = stats.jarque_bera(returns)
    results["Jarque-Bera"] = {
        "Statistique": jb_stat,
        "p-value": jb_pvalue,
        "Conclusion": "Normale" if jb_pvalue > alpha else "Non normale",
        "Interprétation": f"p-value = {jb_pvalue:.4f} {'>' if jb_pvalue > alpha else '≤'} {alpha}"
    }
    
    # Test de Kolmogorov-Smirnov
    ks_stat, ks_pvalue = stats.kstest(returns, 'norm', args=(returns.mean(), returns.std()))
    results["Kolmogorov-Smirnov"] = {
        "Statistique": ks_stat,
        "p-value": ks_pvalue,
        "Conclusion": "Normale" if ks_pvalue > alpha else "Non normale",
        "Interprétation": f"p-value = {ks_pvalue:.4f} {'>' if ks_pvalue > alpha else '≤'} {alpha}"
    }
    
    return results


def qq_plot_data(returns: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prépare les données pour un QQ-plot (Quantile-Quantile plot)
    
    Le QQ-plot compare les quantiles de la distribution empirique
    avec les quantiles d'une distribution normale théorique.
    
    Args:
        returns: Série de rendements
    
    Returns:
        Tuple (theoretical_quantiles, sample_quantiles)
    
    Interprétation:
        - Si les points sont alignés sur la diagonale: distribution normale
        - Écarts en queue: distribution non normale
    """
    # Standardiser les rendements
    standardized = (returns - returns.mean()) / returns.std()
    
    # Calculer les quantiles théoriques et empiriques
    theoretical_quantiles = stats.probplot(standardized, dist="norm")[0][0]
    sample_quantiles = stats.probplot(standardized, dist="norm")[0][1]
    
    return theoretical_quantiles, sample_quantiles


def distribution_analysis(returns: pd.Series) -> Dict[str, any]:
    """
    Analyse complète de la distribution des rendements
    
    Args:
        returns: Série de rendements
    
    Returns:
        Dictionnaire avec toutes les analyses
    """
    analysis = {
        "Statistiques descriptives": descriptive_stats(returns),
        "Moments d'ordre supérieur": higher_moments(returns),
        "Tests de normalité": normality_test(returns)
    }
    
    return analysis


def value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calcule la Value at Risk (VaR) historique
    
    La VaR représente la perte maximale attendue avec un certain niveau
    de confiance sur une période donnée.
    
    Args:
        returns: Série de rendements
        confidence_level: Niveau de confiance (par défaut 95%)
    
    Returns:
        VaR (valeur positive représentant une perte)
    
    Interprétation:
        VaR(95%) = 0.02 signifie qu'il y a 5% de chances de perdre
        plus de 2% sur la période considérée.
    """
    var = -returns.quantile(1 - confidence_level)
    return var


def conditional_value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calcule la Conditional Value at Risk (CVaR) ou Expected Shortfall
    
    La CVaR représente la perte moyenne au-delà de la VaR.
    
    Args:
        returns: Série de rendements
        confidence_level: Niveau de confiance (par défaut 95%)
    
    Returns:
        CVaR (valeur positive représentant une perte)
    """
    var = value_at_risk(returns, confidence_level)
    cvar = -returns[returns <= -var].mean()
    return cvar


def rolling_statistics(returns: pd.Series, window: int = 20) -> pd.DataFrame:
    """
    Calcule des statistiques glissantes
    
    Args:
        returns: Série de rendements
        window: Taille de la fenêtre glissante
    
    Returns:
        DataFrame avec les statistiques glissantes
    """
    rolling_stats = pd.DataFrame({
        "Moyenne": returns.rolling(window).mean(),
        "Écart-type": returns.rolling(window).std(),
        "Min": returns.rolling(window).min(),
        "Max": returns.rolling(window).max()
    })
    
    return rolling_stats
