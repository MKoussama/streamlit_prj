"""
Plateforme d'Analyse Financière
Application Streamlit professionnelle inspirée de TradingView et Bloomberg

Auteur: Projet de Mathématiques Appliquées à la Finance
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Ajouter le dossier modules au path
sys.path.append(os.path.dirname(__file__))

# Importer les modules
from modules.data_loader import load_from_yahoo, load_from_csv, validate_data, get_data_info
from modules.math_operations import (
    arithmetic_returns, log_returns, mean_returns, variance_returns,
    volatility_annualized, correlation_matrix, cumulative_returns
)
from modules.statistics import (
    descriptive_stats, higher_moments, normality_test, 
    qq_plot_data, distribution_analysis, value_at_risk
)
from modules.technical_indicators import (
    SMA, EMA, RSI, bollinger_bands, MACD, ATR, get_all_indicators
)
from modules.backtesting import (
    backtest_sma_crossover, compare_with_buy_and_hold
)
from modules.visualizations import (
    plot_price_with_indicators, plot_candlestick, plot_volume,
    plot_returns_distribution, plot_qq_plot, plot_cumulative_returns,
    plot_backtest_results, plot_drawdown, plot_rsi_subplot, plot_macd_subplot
)
import config

# Configuration de la page
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("📊 Plateforme d'Analyse Financière")
st.markdown("*Application de Mathématiques Appliquées à la Finance*")
st.markdown("---")

# ============================================================================
# SECTION 1 : EN-TÊTE ET CONFIGURATION
# ============================================================================

st.sidebar.header("⚙ Configuration")

# Sélection de la source de données
data_source = st.sidebar.radio(
    "Source de données",
    ["Yahoo Finance", "Fichier CSV"]
)

# Sélection de l'actif
if data_source == "Yahoo Finance":
    # Catégories d'actifs
    category = st.sidebar.selectbox(
        "Catégorie d'actif",
        list(config.PREDEFINED_TICKERS.keys())
    )
    
    ticker = st.sidebar.selectbox(
        "Actif financier",
        config.PREDEFINED_TICKERS[category]
    )
    
    # Option pour un ticker personnalisé
    custom_ticker = st.sidebar.text_input("Ou entrez un ticker personnalisé")
    if custom_ticker:
        ticker = custom_ticker
    
    # Sélection de la période
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Date de début",
            value=datetime.now() - timedelta(days=365)
        )
    with col2:
        end_date = st.date_input(
            "Date de fin",
            value=datetime.now()
        )
    
    # Sélection de l'intervalle
    interval = st.sidebar.selectbox(
        "Fréquence",
        list(config.TIME_INTERVALS.keys())
    )
    interval_code = config.TIME_INTERVALS[interval]

else:
    uploaded_file = st.sidebar.file_uploader(
        "Charger un fichier CSV",
        type=['csv']
    )

# Bouton de chargement
load_button = st.sidebar.button("↻ Charger les données", type="primary")

# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================

if load_button or 'data' in st.session_state:
    
    if load_button:
        with st.spinner("Chargement des données..."):
            try:
                if data_source == "Yahoo Finance":
                    data = load_from_yahoo(
                        ticker, 
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                        interval_code
                    )
                    st.session_state['ticker'] = ticker
                else:
                    if uploaded_file is not None:
                        data = load_from_csv(uploaded_file)
                        st.session_state['ticker'] = "CSV Data"
                    else:
                        st.error("Veuillez charger un fichier CSV")
                        st.stop()
                
                # Valider les données
                is_valid, message = validate_data(data)
                if not is_valid:
                    st.error(f"Erreur de validation: {message}")
                    st.stop()
                
                # Stocker dans la session
                st.session_state['data'] = data
                st.success(f"✓ Données chargées avec succès! {len(data)} périodes")
                
            except Exception as e:
                st.error(f"Erreur lors du chargement: {str(e)}")
                st.stop()
    
    # Récupérer les données de la session
    data = st.session_state['data']
    ticker = st.session_state.get('ticker', 'Actif')
    
    # Informations sur les données
    with st.expander("ℹ Informations sur les données"):
        info = get_data_info(data)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Périodes", info['nombre_periodes'])
        col2.metric("Prix Min", f"${info['prix_min']:.2f}")
        col3.metric("Prix Max", f"${info['prix_max']:.2f}")
        col4.metric("Volume Total", f"{info['volume_total']:,.0f}")
    
    # ============================================================================
    # SECTION 2 : GRAPHIQUE PRINCIPAL
    # ============================================================================
    
    st.header("📈 Graphique Principal")
    
    # Options de visualisation
    col1, col2 = st.columns([3, 1])
    
    with col1:
        chart_type = st.radio(
            "Type de graphique",
            ["Ligne", "Chandeliers"],
            horizontal=True
        )
    
    with col2:
        show_volume = st.checkbox("Afficher le volume", value=True)
    
    # Sélection des indicateurs
    st.subheader("Indicateurs Techniques")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        show_sma = st.checkbox("SMA", value=True)
        if show_sma:
            sma_short = st.number_input("SMA courte", value=20, min_value=1)
            sma_long = st.number_input("SMA longue", value=50, min_value=1)
    
    with col2:
        show_ema = st.checkbox("EMA")
        if show_ema:
            ema_period = st.number_input("Période EMA", value=12, min_value=1)
    
    with col3:
        show_bollinger = st.checkbox("Bandes de Bollinger")
        if show_bollinger:
            bb_period = st.number_input("Période BB", value=20, min_value=1)
            bb_std = st.number_input("Écarts-types", value=2.0, min_value=0.1)
    
    with col4:
        show_rsi = st.checkbox("RSI")
        show_macd = st.checkbox("MACD")
    
    # Calculer les indicateurs sélectionnés
    data_with_indicators = data.copy()
    indicators_to_plot = []
    
    if show_sma:
        data_with_indicators[f'SMA_{sma_short}'] = SMA(data['Close'], sma_short)
        data_with_indicators[f'SMA_{sma_long}'] = SMA(data['Close'], sma_long)
        indicators_to_plot.extend([f'SMA_{sma_short}', f'SMA_{sma_long}'])
    
    if show_ema:
        data_with_indicators[f'EMA_{ema_period}'] = EMA(data['Close'], ema_period)
        indicators_to_plot.append(f'EMA_{ema_period}')
    
    if show_bollinger:
        upper, middle, lower = bollinger_bands(data['Close'], bb_period, bb_std)
        data_with_indicators['BB_Upper'] = upper
        data_with_indicators['BB_Middle'] = middle
        data_with_indicators['BB_Lower'] = lower
        indicators_to_plot.extend(['BB_Upper', 'BB_Middle', 'BB_Lower'])
    
    # Afficher le graphique principal
    if chart_type == "Ligne":
        fig = plot_price_with_indicators(
            data_with_indicators,
            indicators_to_plot,
            title=f"{ticker} - Prix et Indicateurs",
            height=600
        )
    else:
        fig = plot_candlestick(
            data_with_indicators,
            indicators_to_plot,
            title=f"{ticker} - Chandeliers",
            height=600
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Afficher le volume
    if show_volume:
        fig_volume = plot_volume(data_with_indicators, height=200)
        st.plotly_chart(fig_volume, use_container_width=True)
    
    # ============================================================================
    # SECTION 3 : STATISTIQUES ET ANALYSES
    # ============================================================================
    
    st.header("📉 Analyses Statistiques")
    
    # Calculer les rendements
    returns_type = st.radio(
        "Type de rendements",
        ["Arithmétiques", "Logarithmiques"],
        horizontal=True,
        help=config.HELP_MESSAGES['rendements_arithmetiques']
    )
    
    if returns_type == "Arithmétiques":
        returns = arithmetic_returns(data['Close'])
    else:
        returns = log_returns(data['Close'])
    
    # Onglets pour différentes analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Statistiques Descriptives",
        "📉 Distribution",
        "🔬 Tests de Normalité",
        "⚠ Risque (VaR)"
    ])
    
    with tab1:
        st.subheader("Statistiques Descriptives")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Statistiques de base**")
            stats = descriptive_stats(returns)
            stats_df = pd.DataFrame(stats.items(), columns=['Métrique', 'Valeur'])
            st.dataframe(stats_df, hide_index=True)
        
        with col2:
            st.markdown("**Moments d'ordre supérieur**")
            moments = higher_moments(returns)
            st.metric("Skewness", f"{moments['Skewness']:.4f}")
            st.caption(moments['Interprétation Skewness'])
            st.metric("Kurtosis", f"{moments['Kurtosis']:.4f}")
            st.caption(moments['Interprétation Kurtosis'])
        
        # Volatilité annualisée
        vol_annual = volatility_annualized(returns, config.TRADING_DAYS_PER_YEAR)
        st.metric(
            "Volatilité Annualisée",
            f"{vol_annual*100:.2f}%",
            help=config.HELP_MESSAGES['volatilite']
        )
    
    with tab2:
        st.subheader("Distribution des Rendements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogramme
            fig_hist = plot_returns_distribution(returns)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # QQ-Plot
            fig_qq = plot_qq_plot(returns)
            st.plotly_chart(fig_qq, use_container_width=True)
        
        # Rendements cumulés
        fig_cumul = plot_cumulative_returns(returns)
        st.plotly_chart(fig_cumul, use_container_width=True)
    
    with tab3:
        st.subheader("Tests de Normalité")
        
        tests = normality_test(returns)
        
        for test_name, test_results in tests.items():
            with st.expander(f"📋 {test_name}"):
                col1, col2 = st.columns(2)
                col1.metric("Statistique", f"{test_results['Statistique']:.4f}")
                col2.metric("p-value", f"{test_results['p-value']:.4f}")
                
                if test_results['Conclusion'] == "Normale":
                    st.success(f"✓ {test_results['Interprétation']}")
                else:
                    st.warning(f"⚠ {test_results['Interprétation']}")
                
                st.info(
                    "**Interprétation**: Si p-value > 0.05, on ne peut pas rejeter "
                    "l'hypothèse de normalité. Si p-value ≤ 0.05, les données ne "
                    "suivent probablement pas une distribution normale."
                )
    
    with tab4:
        st.subheader("Value at Risk (VaR)")
        
        confidence_level = st.slider(
            "Niveau de confiance",
            min_value=0.90,
            max_value=0.99,
            value=0.95,
            step=0.01
        )
        
        var = value_at_risk(returns, confidence_level)
        
        st.metric(
            f"VaR ({confidence_level*100:.0f}%)",
            f"{var*100:.2f}%"
        )
        
        st.info(
            f"Il y a {(1-confidence_level)*100:.0f}% de chances de perdre "
            f"plus de {var*100:.2f}% sur une période."
        )
    
    # ============================================================================
    # SECTION 4 : INDICATEURS TECHNIQUES (SOUS-GRAPHIQUES)
    # ============================================================================
    
    if show_rsi or show_macd:
        st.header("📊 Indicateurs Techniques")
        
        if show_rsi:
            rsi_period = st.slider("Période RSI", min_value=5, max_value=30, value=14)
            rsi_values = RSI(data['Close'], rsi_period)
            fig_rsi = plot_rsi_subplot(rsi_values)
            st.plotly_chart(fig_rsi, use_container_width=True)
            
            # Valeur actuelle du RSI
            current_rsi = rsi_values.iloc[-1]
            if current_rsi > 70:
                st.warning(f"⚠ RSI actuel: {current_rsi:.2f} - Sur-acheté")
            elif current_rsi < 30:
                st.success(f"✓ RSI actuel: {current_rsi:.2f} - Sur-vendu")
            else:
                st.info(f"ℹ RSI actuel: {current_rsi:.2f} - Neutre")
        
        if show_macd:
            macd, signal, histogram = MACD(data['Close'])
            fig_macd = plot_macd_subplot(macd, signal, histogram)
            st.plotly_chart(fig_macd, use_container_width=True)
    
    # ============================================================================
    # SECTION 5 : BACKTESTING
    # ============================================================================
    
    st.header("🔬 Backtesting de Stratégie")
    
    st.markdown("""
    **Stratégie implémentée**: Croisement de moyennes mobiles simples (SMA)
    - **Signal d'achat**: Lorsque la SMA courte dépasse la SMA longue
    - **Signal de vente**: Lorsque la SMA courte passe sous la SMA longue
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        bt_sma_short = st.number_input(
            "SMA courte (backtest)",
            value=20,
            min_value=1,
            key="bt_sma_short"
        )
    
    with col2:
        bt_sma_long = st.number_input(
            "SMA longue (backtest)",
            value=50,
            min_value=1,
            key="bt_sma_long"
        )
    
    with col3:
        initial_capital = st.number_input(
            "Capital initial ($)",
            value=config.DEFAULT_INITIAL_CAPITAL,
            min_value=100
        )
    
    with col4:
        transaction_fee = st.number_input(
            "Frais de transaction (%)",
            value=config.DEFAULT_TRANSACTION_FEE * 100,
            min_value=0.0,
            max_value=5.0,
            step=0.1
        ) / 100
    
    if st.button("▶ Lancer le Backtest", type="primary"):
        with st.spinner("Exécution du backtest..."):
            # Exécuter le backtest
            backtest_results = backtest_sma_crossover(
                data,
                bt_sma_short,
                bt_sma_long,
                initial_capital,
                transaction_fee
            )
            
            # Stocker dans la session
            st.session_state['backtest_results'] = backtest_results
    
    # Afficher les résultats si disponibles
    if 'backtest_results' in st.session_state:
        results = st.session_state['backtest_results']
        
        # Graphique des résultats
        fig_backtest = plot_backtest_results(
            results['data'],
            title=f"Backtest - {ticker}"
        )
        st.plotly_chart(fig_backtest, use_container_width=True)
        
        # Métriques de performance
        st.subheader("📊 Métriques de Performance")
        
        metrics = results['metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric(
            "Rendement Total",
            f"{metrics['Rendement total']*100:.2f}%"
        )
        col2.metric(
            "Rendement Annualisé",
            f"{metrics['Rendement annualisé']*100:.2f}%"
        )
        col3.metric(
            "Volatilité Annualisée",
            f"{metrics['Volatilité annualisée']*100:.2f}%"
        )
        col4.metric(
            "Ratio de Sharpe",
            f"{metrics['Ratio de Sharpe']:.2f}",
            help=config.HELP_MESSAGES['sharpe']
        )
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric(
            "Maximum Drawdown",
            f"{metrics['Maximum Drawdown']*100:.2f}%",
            help=config.HELP_MESSAGES['drawdown']
        )
        col2.metric(
            "Profit Factor",
            f"{metrics['Profit Factor']:.2f}"
        )
        col3.metric(
            "Nombre de Trades",
            f"{metrics['Nombre de trades']}"
        )
        col4.metric(
            "Taux de Réussite",
            f"{metrics['Taux de réussite']*100:.1f}%"
        )
        
        # Graphique du drawdown
        fig_dd = plot_drawdown(results['data']['Capital'])
        st.plotly_chart(fig_dd, use_container_width=True)
        
        # Comparaison avec Buy & Hold
        st.subheader("📈 Comparaison avec Buy & Hold")
        
        comparison = compare_with_buy_and_hold(
            results['data']['Capital'],
            data['Close'],
            initial_capital
        )
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric(
            "Stratégie SMA",
            f"${comparison['Strategy Capital']:.2f}",
            f"{comparison['Strategy Return']*100:.2f}%"
        )
        col2.metric(
            "Buy & Hold",
            f"${comparison['Buy & Hold Capital']:.2f}",
            f"{comparison['Buy & Hold Return']*100:.2f}%"
        )
        col3.metric(
            "Alpha (Surperformance)",
            f"{comparison['Alpha']*100:.2f}%",
            delta=comparison['Outperformance']
        )
        
        # Liste des trades
        if len(results['trades']) > 0:
            with st.expander("📋 Liste des Trades"):
                st.dataframe(results['trades'], use_container_width=True)
    
    # ============================================================================
    # SECTION 6 : DONNÉES BRUTES
    # ============================================================================
    
    with st.expander("📄 Données OHLC"):
        st.dataframe(data.tail(50), use_container_width=True)
    
    # ============================================================================
    # FOOTER
    # ============================================================================
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><strong>Plateforme d'Analyse Financière</strong></p>
        <p>Projet de Mathématiques Appliquées à la Finance</p>
        <p><em>Inspiré de TradingView, Bloomberg et Binance</em></p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Message d'accueil
    st.info("← Configurez les paramètres dans la barre latérale et cliquez sur 'Charger les données' pour commencer")
    
    st.markdown("""
    ## 🎯 Fonctionnalités
    
    Cette plateforme offre des outils d'analyse financière professionnels :
    
    ### 📊 Acquisition de Données
    - Yahoo Finance (actions, indices, crypto, forex)
    - Import de fichiers CSV
    - Validation automatique des données
    
    ### 📈 Analyses Mathématiques
    - Rendements arithmétiques et logarithmiques
    - Statistiques descriptives complètes
    - Moments d'ordre supérieur (Skewness, Kurtosis)
    - Volatilité annualisée
    - Tests de normalité (Shapiro-Wilk, Jarque-Bera)
    - Value at Risk (VaR)
    
    ### 📉 Indicateurs Techniques
    - Moyennes Mobiles (SMA, EMA)
    - RSI (Relative Strength Index)
    - Bandes de Bollinger
    - MACD (Moving Average Convergence Divergence)
    - ATR (Average True Range)
    
    ### 🔬 Backtesting
    - Stratégie de croisement de moyennes mobiles
    - Métriques de performance complètes
    - Ratio de Sharpe, Maximum Drawdown, Profit Factor
    - Comparaison avec Buy & Hold
    - Liste détaillée des trades
    
    ### 📊 Visualisations
    - Graphiques interactifs (Plotly)
    - Chandeliers japonais
    - Histogrammes et QQ-plots
    - Graphiques de drawdown
    - Sous-graphiques pour RSI et MACD
    """)
