"""
Module de visualisations professionnelles
Graphiques inspirés de TradingView et Bloomberg
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from typing import Optional, List


def plot_price_with_indicators(data: pd.DataFrame, 
                               indicators: List[str] = None,
                               title: str = "Prix et Indicateurs Techniques",
                               height: int = 600) -> go.Figure:
    """
    Graphique principal avec prix et indicateurs techniques
    
    Args:
        data: DataFrame avec colonnes OHLC et indicateurs
        indicators: Liste des colonnes d'indicateurs à afficher
        title: Titre du graphique
        height: Hauteur du graphique
    
    Returns:
        Figure Plotly
    """
    fig = go.Figure()
    
    # Ajouter le prix de clôture
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Prix',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Ajouter les indicateurs
    if indicators:
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for i, indicator in enumerate(indicators):
            if indicator in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[indicator],
                    mode='lines',
                    name=indicator,
                    line=dict(color=colors[i % len(colors)], width=1.5)
                ))
    
    # Mise en forme
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Prix",
        height=height,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def plot_candlestick(data: pd.DataFrame, 
                    indicators: List[str] = None,
                    title: str = "Graphique en Chandeliers",
                    height: int = 600) -> go.Figure:
    """
    Graphique en chandeliers (candlestick) avec indicateurs
    
    Args:
        data: DataFrame OHLC
        indicators: Liste des indicateurs à superposer
        title: Titre du graphique
        height: Hauteur du graphique
    
    Returns:
        Figure Plotly
    """
    fig = go.Figure()
    
    # Ajouter les chandeliers
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='OHLC'
    ))
    
    # Ajouter les indicateurs
    if indicators:
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, indicator in enumerate(indicators):
            if indicator in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[indicator],
                    mode='lines',
                    name=indicator,
                    line=dict(color=colors[i % len(colors)], width=1.5)
                ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Prix",
        height=height,
        xaxis_rangeslider_visible=False,
        template='plotly_white'
    )
    
    return fig


def plot_volume(data: pd.DataFrame, height: int = 200) -> go.Figure:
    """
    Graphique de volume
    
    Args:
        data: DataFrame avec colonne Volume
        height: Hauteur du graphique
    
    Returns:
        Figure Plotly
    """
    # Déterminer les couleurs (vert si hausse, rouge si baisse)
    colors = ['green' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'red' 
              for i in range(len(data))]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume',
        marker_color=colors
    ))
    
    fig.update_layout(
        title="Volume",
        xaxis_title="Date",
        yaxis_title="Volume",
        height=height,
        template='plotly_white',
        showlegend=False
    )
    
    return fig


def plot_returns_distribution(returns: pd.Series, 
                              bins: int = 50,
                              title: str = "Distribution des Rendements") -> go.Figure:
    """
    Histogramme des rendements avec courbe de densité
    
    Args:
        returns: Série de rendements
        bins: Nombre de bins
        title: Titre du graphique
    
    Returns:
        Figure Plotly
    """
    fig = go.Figure()
    
    # Histogramme
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=bins,
        name='Rendements',
        histnorm='probability density',
        marker_color='#1f77b4',
        opacity=0.7
    ))
    
    # Courbe de densité normale théorique
    x_range = np.linspace(returns.min(), returns.max(), 100)
    normal_density = scipy_stats.norm.pdf(x_range, returns.mean(), returns.std())
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=normal_density,
        mode='lines',
        name='Distribution Normale',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Rendements",
        yaxis_title="Densité",
        template='plotly_white',
        showlegend=True
    )
    
    return fig


def plot_qq_plot(returns: pd.Series, title: str = "QQ-Plot") -> go.Figure:
    """
    QQ-Plot pour tester la normalité
    
    Args:
        returns: Série de rendements
        title: Titre du graphique
    
    Returns:
        Figure Plotly
    """
    # Standardiser les rendements
    standardized = (returns - returns.mean()) / returns.std()
    
    # Calculer les quantiles
    theoretical_quantiles, sample_quantiles = scipy_stats.probplot(standardized, dist="norm")
    
    fig = go.Figure()
    
    # Points du QQ-plot
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles[0],
        y=theoretical_quantiles[1],
        mode='markers',
        name='Quantiles observés',
        marker=dict(color='#1f77b4', size=5)
    ))
    
    # Ligne de référence (y = x)
    min_val = min(theoretical_quantiles[0].min(), theoretical_quantiles[1].min())
    max_val = max(theoretical_quantiles[0].max(), theoretical_quantiles[1].max())
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Distribution normale',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Quantiles théoriques",
        yaxis_title="Quantiles observés",
        template='plotly_white'
    )
    
    return fig


def plot_cumulative_returns(returns: pd.Series, 
                           title: str = "Rendements Cumulés") -> go.Figure:
    """
    Graphique des rendements cumulés
    
    Args:
        returns: Série de rendements
        title: Titre du graphique
    
    Returns:
        Figure Plotly
    """
    cumulative = (1 + returns).cumprod() - 1
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=cumulative.index,
        y=cumulative * 100,  # En pourcentage
        mode='lines',
        name='Rendements cumulés',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Rendements cumulés (%)",
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def plot_backtest_results(data: pd.DataFrame, 
                         title: str = "Résultats du Backtesting") -> go.Figure:
    """
    Graphique complet des résultats de backtesting
    
    Args:
        data: DataFrame avec colonnes Close, SMA_short, SMA_long, Position, Capital
        title: Titre du graphique
    
    Returns:
        Figure Plotly avec 2 sous-graphiques
    """
    # Créer des sous-graphiques
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Prix et Signaux de Trading', 'Évolution du Capital'),
        row_heights=[0.6, 0.4]
    )
    
    # Graphique 1: Prix et SMA
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Close'], mode='lines', 
                  name='Prix', line=dict(color='#1f77b4', width=2)),
        row=1, col=1
    )
    
    # Ajouter les SMA si disponibles
    sma_cols = [col for col in data.columns if 'SMA_' in col]
    colors = ['#ff7f0e', '#2ca02c']
    for i, col in enumerate(sma_cols[:2]):
        fig.add_trace(
            go.Scatter(x=data.index, y=data[col], mode='lines',
                      name=col, line=dict(color=colors[i], width=1.5)),
            row=1, col=1
        )
    
    # Ajouter les signaux d'achat/vente
    if 'Position' in data.columns:
        # Signaux d'achat (0 -> 1)
        buy_signals = data[data['Position'].diff() == 1]
        fig.add_trace(
            go.Scatter(x=buy_signals.index, y=buy_signals['Close'],
                      mode='markers', name='Achat',
                      marker=dict(symbol='triangle-up', size=12, color='green')),
            row=1, col=1
        )
        
        # Signaux de vente (1 -> 0)
        sell_signals = data[data['Position'].diff() == -1]
        fig.add_trace(
            go.Scatter(x=sell_signals.index, y=sell_signals['Close'],
                      mode='markers', name='Vente',
                      marker=dict(symbol='triangle-down', size=12, color='red')),
            row=1, col=1
        )
    
    # Graphique 2: Évolution du capital
    if 'Capital' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Capital'], mode='lines',
                      name='Capital', line=dict(color='#2ca02c', width=2),
                      fill='tozeroy', fillcolor='rgba(44, 160, 44, 0.2)'),
            row=2, col=1
        )
    
    # Mise en forme
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Prix", row=1, col=1)
    fig.update_yaxes(title_text="Capital", row=2, col=1)
    
    fig.update_layout(
        title=title,
        height=800,
        template='plotly_white',
        hovermode='x unified',
        showlegend=True
    )
    
    return fig


def plot_drawdown(capital: pd.Series, title: str = "Drawdown") -> go.Figure:
    """
    Graphique du drawdown
    
    Args:
        capital: Série d'évolution du capital
        title: Titre du graphique
    
    Returns:
        Figure Plotly
    """
    # Calculer le drawdown
    running_max = capital.expanding().max()
    drawdown = (capital - running_max) / running_max * 100  # En pourcentage
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown,
        mode='lines',
        name='Drawdown',
        line=dict(color='red', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.2)'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def plot_rsi_subplot(rsi: pd.Series, title: str = "RSI") -> go.Figure:
    """
    Graphique du RSI avec zones de sur-achat/sur-vente
    
    Args:
        rsi: Série de RSI
        title: Titre du graphique
    
    Returns:
        Figure Plotly
    """
    fig = go.Figure()
    
    # RSI
    fig.add_trace(go.Scatter(
        x=rsi.index,
        y=rsi,
        mode='lines',
        name='RSI',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Lignes de référence
    fig.add_hline(y=70, line_dash="dash", line_color="red", 
                  annotation_text="Sur-acheté (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", 
                  annotation_text="Sur-vendu (30)")
    fig.add_hline(y=50, line_dash="dot", line_color="gray")
    
    # Zone de sur-achat (70-100)
    fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, line_width=0)
    
    # Zone de sur-vente (0-30)
    fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, line_width=0)
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="RSI",
        yaxis_range=[0, 100],
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def plot_macd_subplot(macd: pd.Series, signal: pd.Series, histogram: pd.Series,
                     title: str = "MACD") -> go.Figure:
    """
    Graphique du MACD avec ligne de signal et histogramme
    
    Args:
        macd: Série MACD
        signal: Série de la ligne de signal
        histogram: Série de l'histogramme
        title: Titre du graphique
    
    Returns:
        Figure Plotly
    """
    fig = go.Figure()
    
    # MACD
    fig.add_trace(go.Scatter(
        x=macd.index,
        y=macd,
        mode='lines',
        name='MACD',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Signal
    fig.add_trace(go.Scatter(
        x=signal.index,
        y=signal,
        mode='lines',
        name='Signal',
        line=dict(color='#ff7f0e', width=2)
    ))
    
    # Histogramme
    colors = ['green' if val >= 0 else 'red' for val in histogram]
    fig.add_trace(go.Bar(
        x=histogram.index,
        y=histogram,
        name='Histogramme',
        marker_color=colors,
        opacity=0.5
    ))
    
    # Ligne zéro
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="MACD",
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, 
                             title: str = "Matrice de Corrélation") -> go.Figure:
    """
    Heatmap de la matrice de corrélation
    
    Args:
        corr_matrix: Matrice de corrélation
        title: Titre du graphique
    
    Returns:
        Figure Plotly
    """
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Corrélation")
    ))
    
    fig.update_layout(
        title=title,
        template='plotly_white',
        width=600,
        height=600
    )
    
    return fig
