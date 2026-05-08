import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Configuração da Página para Estilo Dark Moderno
st.set_page_config(page_title="Quant Lab Dash", layout="wide")

# CSS para criar os Cards arredondados iguais à sua foto
st.markdown("""
    <style>
    .asset-card {
        background-color: #161a25;
        border: 1px solid #2a2e39;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 10px;
    }
    .stApp { background-color: #0b0e14; }
    </style>
    """, unsafe_allow_html=True)

# Menu Lateral conforme sua foto
st.sidebar.title("OM Quant Services")
st.sidebar.button("🏠 Boas-vindas")
st.sidebar.subheader("Modelos")

# Título Principal
st.title("Painel de Simulações Backtest")

# Exemplo de como os ativos aparecerão (ETHUSDT)
with st.container():
    st.markdown('<div class="asset-card">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([2, 4, 2, 1])
    
    with col1:
        st.subheader("ETHUSDT 🔵")
        st.write("Price: 2,290.48")
    
    with col2:
        # Aqui você ligaria o seu gráfico de backtest do Colab
        st.line_chart([10, 12, 11, 15, 14, 13], height=100)
    
    with col3:
        st.metric("Profit", "+12.5%", "1.2%")
        
    with col4:
        st.button("+", key="eth")
    st.markdown('</div>', unsafe_allow_html=True)

st.button("➕ Adicionar Ativo")
