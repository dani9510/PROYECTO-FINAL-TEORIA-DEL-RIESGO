import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Riesgo Financiero",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard de Riesgo Financiero")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Configuración")

# Símbolos disponibles
available_symbols = {
    "Acciones Tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"],
    "ETFs": ["SPY", "QQQ", "IWM", "TLT", "GLD", "VTI"],
    "Índices": ["^GSPC", "^IXIC", "^DJI"],
    "Forex": ["EURUSD=X", "GBPUSD=X", "JPY=X"],
    "Crypto": ["BTC-USD", "ETH-USD", "ADA-USD"]
}

selected_category = st.sidebar.selectbox(
    "Categoría:",
    list(available_symbols.keys())
)

selected_symbols = st.sidebar.multiselect(
    "Selecciona activos:",
    options=available_symbols[selected_category],
    default=["AAPL", "MSFT", "GOOGL"] if selected_category == "Acciones Tech" else available_symbols[selected_category][:3]
)

# Configuración de fechas
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Fecha inicio:",
        datetime.now() - timedelta(days=365*3)
    )
with col2:
    end_date = st.date_input("Fecha fin:", datetime.now())

confidence_level = st.sidebar.slider(
    "Nivel de confianza VaR (%):",
    min_value=90,
    max_value=99,
    value=95
) / 100

st.sidebar.markdown("---")
st.sidebar.info("💡 Los datos se descargan de Yahoo Finance")

# Función para descargar datos
@st.cache_data(ttl=3600)
def download_data(symbols_list, start_dt, end_dt):
    price_data = {}
    returns_data = {}
    successful_downloads = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, symbol in enumerate(symbols_list):
        status_text.text(f"Descargando {symbol}... ({i+1}/{len(symbols_list)})")
        
        try:
            # Descargar datos
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_dt, end=end_dt, auto_adjust=True)
            
            if data.empty or len(data) < 10:
                st.warning(f"No hay datos suficientes para {symbol}")
                continue
            
            if "Close" not in data.columns:
                st.warning(f"No se encontraron precios para {symbol}")
                continue
            
            price_data[symbol] = data["Close"]
            
            # Calcular rendimientos logarítmicos
            returns = np.log(price_data[symbol] / price_data[symbol].shift(1)).dropna()
            
            if len(returns) > 0:
                returns_data[symbol] = returns
                successful_downloads.append(symbol)
                st.success(f"{symbol} descargado correctamente")
            else:
                st.warning(f"No se pudieron calcular rendimientos para {symbol}")
                
        except Exception as e:
            st.error(f"Error descargando {symbol}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(symbols_list))
    
    progress_bar.empty()
    status_text.text("Descarga completada")
    
    if not returns_data:
        st.error("No se pudieron descargar datos para ningún símbolo")
        return {}, pd.DataFrame()
    
    # Crear DataFrame de rendimientos
    returns_df = pd.DataFrame(returns_data)
    returns_df = returns_df.dropna()
    
    st.success(f"{len(successful_downloads)} de {len(symbols_list)} activos descargados exitosamente")
    return price_data, returns_df

# Función para calcular métricas de riesgo
def calculate_risk_metrics(returns_df, conf_level):
    metrics = {}
    
    for symbol in returns_df.columns:
        returns = returns_df[symbol].dropna()
        
        # Métricas básicas
        media = returns.mean()
        volatilidad = returns.std()
        volatilidad_anual = volatilidad * np.sqrt(252)
        
        # Value at Risk (VaR)
        var_parametrico = abs(-np.percentile(returns, (1 - conf_level) * 100))
        var_historico = abs(returns.quantile(1 - conf_level))
        
        # Expected Shortfall (CVaR)
        losses_below_var = returns[returns < -var_historico]
        expected_shortfall = -losses_below_var.mean() if len(losses_below_var) > 0 else var_historico
        
        # Ratio Sharpe
        sharpe_ratio = media / volatilidad * np.sqrt(252) if volatilidad > 0 else 0
        
        # Máximo Drawdown
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        # Estadísticas adicionales
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        metrics[symbol] = {
            "Media_Diaria": media,
            "Volatilidad_Diaria": volatilidad,
            "Volatilidad_Anual": volatilidad_anual,
            f"VaR_{int(conf_level*100)}%": var_parametrico,
            f"ES_{int(conf_level*100)}%": expected_shortfall,
            "Ratio_Sharpe": sharpe_ratio,
            "Max_Drawdown": max_drawdown,
            "Asimetria": skewness,
            "Curtosis": kurtosis,
            "Observaciones": len(returns)
        }
    
    return pd.DataFrame(metrics).T

# Interfaz principal
if selected_symbols:
    st.subheader("Descargando datos de mercado...")
    
    with st.spinner("Conectando con Yahoo Finance..."):
        price_data, returns_df = download_data(selected_symbols, start_date, end_date)
    
    if not returns_df.empty:
        st.success(f"Análisis de {len(returns_df.columns)} activos completado")
        
        # Calcular métricas
        metrics_df = calculate_risk_metrics(returns_df, confidence_level)
        
        # Crear pestañas
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Precios y Rendimientos", 
            "⚠️ Análisis de Riesgo", 
            "📊 Estadísticas", 
            "🔍 Comparativa"
        ])
        
        with tab1:
            st.subheader("Evolución de Precios")
            
            # Gráfico de precios
            fig_precios = go.Figure()
            for symbol in returns_df.columns:
                if symbol in price_data:
                    fig_precios.add_trace(go.Scatter(
                        x=price_data[symbol].index,
                        y=price_data[symbol].values,
                        name=symbol,
                        line=dict(width=2)
                    ))
            
            fig_precios.update_layout(
                title="Precios de Cierre",
                xaxis_title="Fecha",
                yaxis_title="Precio (USD)",
                height=500,
                showlegend=True
            )
            st.plotly_chart(fig_precios, use_container_width=True)
            
            # Gráfico de rendimientos acumulados
            st.subheader("Rendimientos Acumulados")
            returns_acumulados = (1 + returns_df).cumprod() - 1
            
            fig_rendimientos = go.Figure()
            for symbol in returns_acumulados.columns:
                fig_rendimientos.add_trace(go.Scatter(
                    x=returns_acumulados.index,
                    y=returns_acumulados[symbol],
                    name=symbol,
                    line=dict(width=2)
                ))
            
            fig_rendimientos.update_layout(
                title="Rendimientos Acumulados",
                xaxis_title="Fecha",
                yaxis_title="Rendimiento Acumulado",
                yaxis_tickformat=".0%",
                height=400
            )
            st.plotly_chart(fig_rendimientos, use_container_width=True)
        
        with tab2:
            st.subheader("Métricas de Riesgo")
            
            # Mostrar métricas principales
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_vol = metrics_df["Volatilidad_Anual"].mean()
                st.metric("Volatilidad Promedio Anual", f"{avg_vol:.2%}")
            
            with col2:
                avg_var = metrics_df[f"VaR_{int(confidence_level*100)}%"].mean()
                st.metric(f"VaR Promedio {int(confidence_level*100)}%", f"{avg_var:.4f}")
            
            with col3:
                worst_drawdown = metrics_df["Max_Drawdown"].min()
                st.metric("Peor Drawdown", f"{worst_drawdown:.2%}")
            
            with col4:
                best_sharpe = metrics_df["Ratio_Sharpe"].max()
                st.metric("Mejor Ratio Sharpe", f"{best_sharpe:.2f}")
            
            # Tabla de métricas detalladas
            st.subheader("Métricas por Activo")
            
            # Formatear la tabla para mostrar
            display_metrics = metrics_df.copy()
            format_rules = {
                "Media_Diaria": "{:.6f}",
                "Volatilidad_Diaria": "{:.4f}",
                "Volatilidad_Anual": "{:.2%}",
                f"VaR_{int(confidence_level*100)}%": "{:.4f}",
                f"ES_{int(confidence_level*100)}%": "{:.4f}",
                "Ratio_Sharpe": "{:.2f}",
                "Max_Drawdown": "{:.2%}",
                "Asimetria": "{:.3f}",
                "Curtosis": "{:.3f}"
            }
            
            for col, fmt in format_rules.items():
                if col in display_metrics.columns:
                    display_metrics[col] = display_metrics[col].apply(lambda x: fmt.format(x))
            
            st.dataframe(display_metrics, use_container_width=True)
            
            # Gráfico de VaR comparativo
            st.subheader(f"Comparación de VaR {int(confidence_level*100)}%")
            var_col = f"VaR_{int(confidence_level*100)}%"
            var_data = metrics_df[var_col].sort_values()
            
            fig_var = px.bar(
                x=var_data.index,
                y=var_data.values,
                title=f"Value at Risk ({int(confidence_level*100)}%) por Activo",
                labels={"x": "Activo", "y": "VaR"}
            )
            st.plotly_chart(fig_var, use_container_width=True)
        
        with tab3:
            st.subheader("Estadísticas Descriptivas")
            
            # Seleccionar activo para análisis detallado
            selected_asset = st.selectbox(
                "Selecciona un activo para análisis detallado:",
                returns_df.columns
            )
            
            if selected_asset:
                returns_asset = returns_df[selected_asset].dropna()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histograma de rendimientos
                    fig_hist = px.histogram(
                        returns_asset, 
                        nbins=50,
                        title=f"Distribución de Rendimientos - {selected_asset}",
                        labels={"value": "Rendimiento Diario"}
                    )
                    
                    # Añadir líneas de referencia
                    var_line = -metrics_df.loc[selected_asset, f"VaR_{int(confidence_level*100)}%"]
                    es_line = -metrics_df.loc[selected_asset, f"ES_{int(confidence_level*100)}%"]
                    
                    fig_hist.add_vline(
                        x=var_line, 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text=f"VaR {int(confidence_level*100)}%"
                    )
                    fig_hist.add_vline(
                        x=es_line, 
                        line_dash="dash", 
                        line_color="orange",
                        annotation_text=f"ES {int(confidence_level*100)}%"
                    )
                    
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # Box plot
                    fig_box = px.box(
                        returns_asset,
                        title=f"Box Plot - {selected_asset}",
                        labels={"value": "Rendimiento Diario"}
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
        
        with tab4:
            st.subheader("Análisis Comparativo")
            
            # Heatmap de correlaciones
            st.subheader("Matriz de Correlaciones")
            corr_matrix = returns_df.corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Correlación entre Activos"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Comparativa de volatilidades
            st.subheader("Comparativa de Volatilidades Anuales")
            vol_data = metrics_df["Volatilidad_Anual"].sort_values(ascending=True)
            
            fig_vol = px.bar(
                x=vol_data.values,
                y=vol_data.index,
                orientation="h",
                title="Volatilidad Anual por Activo",
                labels={"x": "Volatilidad Anual", "y": "Activo"}
            )
            st.plotly_chart(fig_vol, use_container_width=True)
    
    else:
        st.error("No se pudieron obtener datos suficientes para el análisis.")
        
else:
    st.info("Por favor selecciona al menos un activo en la barra lateral para comenzar el análisis")

# Footer
st.markdown("---")
st.markdown("**Dashboard de Análisis de Riesgo Financiero** • Desarrollado con Streamlit")