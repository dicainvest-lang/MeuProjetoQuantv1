# -*- coding: utf-8 -*-
# ============================================================
# MOMENTUM ROTACIONAL B3 — Dashboard Interativo Streamlit
# Estratégia + Sinais de Compra/Venda em Tempo Real
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── Configuração da página ───────────────────────────────────
st.set_page_config(
    page_title="Momentum B3 — Quant Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Dark Mode personalizado ──────────────────────────────
st.markdown("""
<style>
    /* Fundo principal */
    .stApp { background-color: #0b0e14; color: #c9d1d9; }
    .stApp > header { background-color: #0b0e14; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #21262d;
    }

    /* Cards / containers */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #161b22;
        border: 1px solid #21262d !important;
        border-radius: 12px !important;
    }

    /* Métricas */
    [data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 14px 18px;
    }
    [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 12px !important; }
    [data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 22px !important; font-weight: 700; }
    [data-testid="stMetricDelta"] { font-size: 13px !important; }

    /* Botão primário */
    .stButton > button {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 14px;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2ea043, #3fb950);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(46,160,67,0.4);
    }

    /* Selectbox / multiselect */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #1f6feb !important;
        border-radius: 6px !important;
    }

    /* Títulos */
    h1 { color: #e6edf3 !important; font-family: 'JetBrains Mono', monospace; }
    h2, h3 { color: #c9d1d9 !important; }

    /* Sinal cards */
    .signal-buy {
        background: linear-gradient(135deg, #0d2818, #1a4731);
        border: 1px solid #238636;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
    }
    .signal-sell {
        background: linear-gradient(135deg, #2d0f0f, #4a1515);
        border: 1px solid #f85149;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
    }
    .signal-hold {
        background: linear-gradient(135deg, #1a1a0f, #2d2a0f);
        border: 1px solid #d29922;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
    }
    .signal-title { font-size: 18px; font-weight: 700; margin-bottom: 6px; }
    .signal-subtitle { font-size: 13px; color: #8b949e; }
    .signal-value { font-size: 28px; font-weight: 800; }

    /* Tabela de sinais */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Divider */
    hr { border-color: #21262d !important; }

    /* Expander */
    [data-testid="stExpander"] {
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
    }

    /* Status badges */
    .badge-buy { background:#238636; color:white; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:700; }
    .badge-sell { background:#f85149; color:white; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:700; }
    .badge-hold { background:#d29922; color:black; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:700; }
    .badge-neutral { background:#30363d; color:#c9d1d9; padding:3px 10px; border-radius:20px; font-size:11px; }

    /* Auto-refresh info */
    .refresh-badge {
        background:#0d1117;
        border: 1px solid #21262d;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 12px;
        color: #3fb950;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ── Constantes ───────────────────────────────────────────────
TICKERS_DEFAULT = {
    "BOVA11.SA": "Ibovespa ETF",
    "VALE3.SA":  "Vale ON",
    "SMAL11.SA": "Small Caps ETF",
    "IVVB11.SA": "S&P500 Hedge",
    "HASH11.SA": "Cripto ETF",
    "GOLD11.SA": "Ouro ETF",
    "XFIX11.SA": "Imob ETF",
}

ROC_FAST   = 9
ROC_MED    = 21
ROC_SLOW   = 63
VOL_WIN    = 21
RSI_WIN    = 14
SMA50_WIN  = 50
SMA200_WIN = 200
W_FAST, W_MED, W_SLOW = 0.5, 0.3, 0.2
CONF_THRESH = 0.10
BENCH_TICKER = "EWZ"
CAPITAL_INICIAL = 1_000_000.0

# ── Funções auxiliares ───────────────────────────────────────
def roc(series, n):
    return series.pct_change(n)

def rolling_std(series, n):
    return series.pct_change().rolling(n).std()

def rsi(series, n=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(n).mean()
    loss  = (-delta.clip(upper=0)).rolling(n).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def sma(series, n):
    return series.rolling(n).mean()

def macd(series, fast=12, slow=26, signal=9):
    ema_fast   = series.ewm(span=fast).mean()
    ema_slow   = series.ewm(span=slow).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram

def bollinger(series, n=20, k=2):
    mid  = series.rolling(n).mean()
    std  = series.rolling(n).std()
    return mid + k*std, mid, mid - k*std

# ── Download e cache ─────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def baixar_dados(tickers_list, start, end):
    all_t = list(set(tickers_list + [BENCH_TICKER]))
    raw   = yf.download(all_t, start=start, end=end,
                        auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers_list[0])
    raw.dropna(how="all", inplace=True)
    raw.ffill(inplace=True)
    return raw

# ── Cálculo de indicadores ───────────────────────────────────
def calcular_indicadores(prices, valid):
    indicators = {}
    for t in valid:
        if t not in prices.columns:
            continue
        s = prices[t]
        m, sig, hist = macd(s)
        bb_up, bb_mid, bb_low = bollinger(s)
        indicators[t] = {
            "roc_fast":  roc(s, ROC_FAST),
            "roc_med":   roc(s, ROC_MED),
            "roc_slow":  roc(s, ROC_SLOW),
            "vol":       rolling_std(s, VOL_WIN),
            "rsi":       rsi(s, RSI_WIN),
            "sma50":     sma(s, SMA50_WIN),
            "sma200":    sma(s, SMA200_WIN),
            "macd":      m,
            "macd_sig":  sig,
            "macd_hist": hist,
            "bb_up":     bb_up,
            "bb_mid":    bb_mid,
            "bb_low":    bb_low,
        }
    return indicators

# ── Score de momentum ────────────────────────────────────────
def calcular_score(t, date, prices, indicators):
    try:
        fast  = indicators[t]["roc_fast"].loc[date]
        med   = indicators[t]["roc_med"].loc[date]
        slow  = indicators[t]["roc_slow"].loc[date]
        vol   = indicators[t]["vol"].loc[date]
        r     = indicators[t]["rsi"].loc[date]
        price = prices[t].loc[date]
        sma50 = indicators[t]["sma50"].loc[date]
        if any(pd.isna([fast, med, slow, vol, r, price, sma50])):
            return np.nan
        if vol == 0: vol = 1.0
        mom    = (fast * W_FAST) + (med * W_MED) + (slow * W_SLOW)
        risk_m = mom / vol
        trend  = 1.0 if price > sma50 else 0.5
        penalty = 0.9 if (r > 85 or r < 30) else 1.0
        return risk_m * trend * penalty
    except:
        return np.nan

# ── Backtest ─────────────────────────────────────────────────
def run_backtest(prices, indicators, valid):
    bench_sma200 = sma(prices[BENCH_TICKER], SMA200_WIN) if BENCH_TICKER in prices.columns else None
    portfolio_value = CAPITAL_INICIAL
    holding = None
    start_idx = 100
    dates = prices.index[start_idx:]
    portfolio_values = {prices.index[start_idx - 1]: CAPITAL_INICIAL}
    holdings_log = []

    for i, date in enumerate(dates):
        day_prices = {t: prices[t].loc[date] for t in valid
                      if t in prices.columns and not pd.isna(prices[t].loc[date])}

        if holding and holding in day_prices and i > 0:
            prev_val = portfolio_values.get(dates[i-1], CAPITAL_INICIAL)
            prev_price = prices[holding].iloc[start_idx + i - 1] if start_idx + i - 1 < len(prices) else None
            if prev_price and not pd.isna(prev_price) and prev_price != 0:
                ret = (day_prices[holding] / prev_price) - 1
                portfolio_value = prev_val * (1 + ret)

        scores = {}
        for t in valid:
            sc = calcular_score(t, date, prices, indicators)
            if not pd.isna(sc):
                scores[t] = sc

        if not scores:
            portfolio_values[date] = portfolio_value
            holdings_log.append({"date": date, "holding": holding, "value": portfolio_value})
            continue

        best_asset = max(scores, key=scores.get)
        best_score = scores[best_asset]
        bench_trend = True
        if bench_sma200 is not None and not pd.isna(bench_sma200.loc[date]) and BENCH_TICKER in prices.columns:
            bench_trend = prices[BENCH_TICKER].loc[date] > bench_sma200.loc[date]

        target = holding
        if holding is None:
            target = best_asset if best_score > 0 else "CASH"
        elif holding == "CASH":
            if best_score > 0.02:
                target = best_asset
        else:
            curr_score = scores.get(holding, -999)
            if best_score > curr_score * (1 + CONF_THRESH):
                target = best_asset
            elif curr_score < -0.02:
                target = "CASH"

        if not bench_trend and target != "CASH":
            curr_sc = scores.get(target, -999)
            if curr_sc < 0:
                target = "CASH"

        holding = target if target else "CASH"
        portfolio_values[date] = portfolio_value
        holdings_log.append({"date": date, "holding": holding, "value": portfolio_value})

    equity = pd.Series(portfolio_values)
    holdings_df = pd.DataFrame(holdings_log).set_index("date")
    return equity, holdings_df

# ── Métricas ─────────────────────────────────────────────────
def calcular_metricas(equity, returns, rf=0.05):
    if len(equity) < 2:
        return {}
    total_days = (equity.index[-1] - equity.index[0]).days
    years = max(total_days / 365.25, 0.01)
    cagr       = (equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1
    total_ret  = (equity.iloc[-1] / equity.iloc[0]) - 1
    vol_anual  = returns.std() * np.sqrt(252)
    sharpe     = (returns.mean() * 252 - rf) / (returns.std() * np.sqrt(252)) if vol_anual > 0 else 0
    neg_ret    = returns[returns < 0]
    sortino    = (returns.mean() * 252 - rf) / (neg_ret.std() * np.sqrt(252)) if len(neg_ret) > 0 else 0
    roll_max   = equity.cummax()
    dd         = (equity - roll_max) / roll_max
    max_dd     = dd.min()
    calmar     = cagr / abs(max_dd) if max_dd != 0 else 0
    win_rate   = (returns > 0).mean()
    avg_win    = returns[returns > 0].mean() if (returns > 0).any() else 0
    avg_loss   = returns[returns < 0].mean() if (returns < 0).any() else 0
    plr        = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    return {
        "CAGR": cagr, "Retorno Total": total_ret,
        "Volatilidade Anual": vol_anual, "Índice de Sharpe": sharpe,
        "Proporção de Sortino": sortino, "Drawdown Máximo": max_dd,
        "Índice de Calmar": calmar, "Taxa de Vitórias": win_rate,
        "Vitória Média": avg_win, "Perda Média": avg_loss,
        "Índice Lucro/Prejuízo": plr, "Patrimônio Final": equity.iloc[-1],
    }

# ── Gerador de sinais em tempo real ──────────────────────────
def gerar_sinais(prices, indicators, valid):
    """Gera sinais de compra/venda para o dia atual baseado no algoritmo."""
    sinais = []
    if len(prices) < 2:
        return pd.DataFrame()

    date = prices.index[-1]
    bench_sma200 = sma(prices[BENCH_TICKER], SMA200_WIN) if BENCH_TICKER in prices.columns else None
    bench_trend = True
    if bench_sma200 is not None and not pd.isna(bench_sma200.iloc[-1]):
        bench_trend = prices[BENCH_TICKER].iloc[-1] > bench_sma200.iloc[-1]

    scores_hoje = {}
    for t in valid:
        sc = calcular_score(t, date, prices, indicators)
        if not pd.isna(sc):
            scores_hoje[t] = sc

    if not scores_hoje:
        return pd.DataFrame()

    best = max(scores_hoje, key=scores_hoje.get)
    scores_sorted = sorted(scores_hoje.items(), key=lambda x: x[1], reverse=True)

    for t, score in scores_sorted:
        if t not in prices.columns:
            continue
        price_atual = prices[t].iloc[-1]
        price_ontem = prices[t].iloc[-2] if len(prices) > 1 else price_atual
        var_dia = (price_atual / price_ontem - 1) * 100 if price_ontem != 0 else 0

        rsi_val   = indicators[t]["rsi"].iloc[-1] if not pd.isna(indicators[t]["rsi"].iloc[-1]) else 50
        macd_val  = indicators[t]["macd"].iloc[-1]
        macd_s    = indicators[t]["macd_sig"].iloc[-1]
        sma50_val = indicators[t]["sma50"].iloc[-1]
        sma200_val= indicators[t]["sma200"].iloc[-1]
        bb_up_val = indicators[t]["bb_up"].iloc[-1]
        bb_low_val= indicators[t]["bb_low"].iloc[-1]

        # Lógica de sinal
        is_best     = (t == best)
        above_sma50 = price_atual > sma50_val if not pd.isna(sma50_val) else False
        macd_bull   = macd_val > macd_s if not pd.isna(macd_val) and not pd.isna(macd_s) else False
        rsi_ok      = 30 < rsi_val < 75
        rsi_over    = rsi_val >= 75
        rsi_over_sold = rsi_val <= 30

        if is_best and score > 0.02 and above_sma50 and bench_trend:
            if rsi_over_sold:
                sinal = "COMPRA FORTE"
                cor   = "#3fb950"
                icone = "🚀"
                forca = 5
            elif rsi_ok and macd_bull:
                sinal = "COMPRA"
                cor   = "#2ea043"
                icone = "✅"
                forca = 4
            else:
                sinal = "COMPRA PARCIAL"
                cor   = "#56d364"
                icone = "📈"
                forca = 3
        elif score < -0.02 or rsi_over or not above_sma50:
            if rsi_over:
                sinal = "VENDA"
                cor   = "#f85149"
                icone = "🔴"
                forca = 1
            elif score < -0.05:
                sinal = "VENDA FORTE"
                cor   = "#da3633"
                icone = "⚠️"
                forca = 0
            else:
                sinal = "REDUZIR"
                cor   = "#ff7b72"
                icone = "📉"
                forca = 2
        else:
            sinal = "MANTER"
            cor   = "#d29922"
            icone = "⏸️"
            forca = 3

        # Filtro macro
        if not bench_trend and "COMPRA" in sinal:
            sinal = "AGUARDAR"
            cor   = "#8b949e"
            icone = "⏳"
            forca = 2

        sinais.append({
            "Ticker":     t,
            "Nome":       TICKERS_DEFAULT.get(t, t),
            "Preço":      price_atual,
            "Var. Dia":   var_dia,
            "Score":      score,
            "RSI":        rsi_val,
            "MACD Bull":  macd_bull,
            "Acima SMA50": above_sma50,
            "Sinal":      sinal,
            "Cor":        cor,
            "Ícone":      icone,
            "Força":      forca,
        })

    return pd.DataFrame(sinais).sort_values("Score", ascending=False)

# ═══════════════════════════════════════════════════════════
# ── SIDEBAR ─────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px;'>
        <span style='font-size:32px'>📊</span>
        <h2 style='color:#e6edf3; margin:8px 0 4px; font-size:16px; font-weight:700;'>
            MOMENTUM B3
        </h2>
        <p style='color:#8b949e; font-size:12px; margin:0;'>Quant Dashboard v2.0</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Configurações")

    # Seleção de ativos
    tickers_selecionados = st.multiselect(
        "Ativos (B3)",
        options=list(TICKERS_DEFAULT.keys()),
        default=list(TICKERS_DEFAULT.keys()),
        format_func=lambda x: f"{x} — {TICKERS_DEFAULT[x]}",
    )

    # Período
    periodo_map = {
        "6 Meses": ("2024-11-09", datetime.today().strftime("%Y-%m-%d")),
        "1 Ano":   ("2024-05-09", datetime.today().strftime("%Y-%m-%d")),
        "2 Anos":  ("2023-05-09", datetime.today().strftime("%Y-%m-%d")),
        "3 Anos":  ("2022-05-09", datetime.today().strftime("%Y-%m-%d")),
        "5 Anos":  ("2020-05-09", datetime.today().strftime("%Y-%m-%d")),
        "Máximo":  ("2019-01-01", datetime.today().strftime("%Y-%m-%d")),
    }
    periodo = st.selectbox("Período de Análise", list(periodo_map.keys()), index=3)
    start_date, end_date = periodo_map[periodo]

    capital = st.number_input(
        "Capital Inicial (R$)",
        min_value=10_000,
        max_value=100_000_000,
        value=1_000_000,
        step=10_000,
        format="%d"
    )
    CAPITAL_INICIAL = float(capital)

    st.divider()
    executar = st.button("🚀 Executar Backtest", type="primary")

    st.divider()
    # Auto-refresh
    st.markdown("### 🔄 Auto-Refresh")
    refresh_interval = st.selectbox(
        "Intervalo de atualização",
        ["Desativado", "1 minuto", "5 minutos", "15 minutos"],
        index=1,
    )

    refresh_ms = {"Desativado": None, "1 minuto": 60000,
                  "5 minutos": 300000, "15 minutos": 900000}

    if refresh_ms[refresh_interval]:
        from streamlit_autorefresh import st_autorefresh
        count = st_autorefresh(
            interval=refresh_ms[refresh_interval],
            limit=None,
            key="autorefresh"
        )

    # Info
    st.divider()
    st.markdown(f"""
    <div style='font-size:11px; color:#8b949e; line-height:1.8;'>
        <b style='color:#c9d1d9;'>📅 Última atualização</b><br>
        {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}<br><br>
        <b style='color:#c9d1d9;'>⚡ Cache TTL:</b> 60 segundos<br>
        <b style='color:#c9d1d9;'>🏦 Benchmark:</b> EWZ (iShares MSCI Brazil)<br>
        <b style='color:#c9d1d9;'>📐 Modelo:</b> Momentum Rotacional
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# ── HEADER PRINCIPAL ────────────────────────────────────────
# ═══════════════════════════════════════════════════════════
col_t1, col_t2 = st.columns([3, 1])
with col_t1:
    st.markdown("""
    <h1 style='font-size:26px; font-weight:800; color:#e6edf3; margin-bottom:4px;'>
        📈 Momentum Rotacional B3
    </h1>
    <p style='color:#8b949e; font-size:14px; margin-top:0;'>
        Estratégia Quant • Sinais em Tempo Real • Backtest Completo
    </p>
    """, unsafe_allow_html=True)
with col_t2:
    st.markdown(f"""
    <div style='text-align:right; padding-top:14px;'>
        <span class='refresh-badge'>🟢 AO VIVO — {datetime.now().strftime('%H:%M:%S')}</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Verificações ─────────────────────────────────────────────
if not tickers_selecionados:
    st.warning("⚠️ Selecione ao menos um ativo na barra lateral.")
    st.stop()

# ── Carregamento dos dados ───────────────────────────────────
with st.spinner("📥 Carregando dados do Yahoo Finance..."):
    try:
        prices = baixar_dados(tickers_selecionados, start_date, end_date)
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        st.stop()

valid = [t for t in tickers_selecionados
         if t in prices.columns and prices[t].notna().sum() > 100]

if not valid:
    st.error("Nenhum ativo com dados suficientes. Tente outro período.")
    st.stop()

# Calcular indicadores
with st.spinner("⚙️ Calculando indicadores técnicos..."):
    indicators = calcular_indicadores(prices, valid)

# ═══════════════════════════════════════════════════════════
# ── ABA 1: SINAIS | ABA 2: BACKTEST | ABA 3: ANÁLISE TÉCNICA
# ═══════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "🎯 Sinais de Compra/Venda",
    "📊 Backtest & Performance",
    "🔬 Análise Técnica por Ativo"
])

# ═══════════════════════════════════════════════════════════
# TAB 1: SINAIS DE COMPRA/VENDA
# ═══════════════════════════════════════════════════════════
with tab1:
    sinais_df = gerar_sinais(prices, indicators, valid)

    if sinais_df.empty:
        st.warning("Dados insuficientes para gerar sinais.")
    else:
        data_ref = prices.index[-1].strftime('%d/%m/%Y')
        st.markdown(f"""
        <div style='background:#161b22; border:1px solid #21262d; border-radius:10px;
                    padding:14px 20px; margin-bottom:20px;'>
            <span style='color:#3fb950; font-weight:700; font-size:15px;'>
                🎯 Sinais gerados para: {data_ref}
            </span>
            <span style='color:#8b949e; font-size:13px; margin-left:16px;'>
                Baseado no modelo Momentum Rotacional com filtro macro EWZ
            </span>
        </div>
        """, unsafe_allow_html=True)

        # ── SINAL PRINCIPAL (melhor ativo) ───────────────────
        melhor = sinais_df.iloc[0]
        pior   = sinais_df.iloc[-1]

        col_a, col_b, col_c, col_d = st.columns(4)

        with col_a:
            sinal_texto = melhor["Sinal"]
            cor_sinal   = "#3fb950" if "COMPRA" in sinal_texto else "#f85149" if "VENDA" in sinal_texto else "#d29922"
            st.markdown(f"""
            <div style='background:#161b22; border:1px solid {cor_sinal}55;
                        border-left: 4px solid {cor_sinal};
                        border-radius:10px; padding:16px;'>
                <p style='color:#8b949e; font-size:11px; margin:0 0 6px;'>MELHOR OPORTUNIDADE</p>
                <p style='color:#e6edf3; font-size:20px; font-weight:800; margin:0 0 4px;'>{melhor["Ícone"]} {melhor["Ticker"]}</p>
                <p style='color:{cor_sinal}; font-size:14px; font-weight:700; margin:0 0 4px;'>{sinal_texto}</p>
                <p style='color:#8b949e; font-size:12px; margin:0;'>Score: {melhor["Score"]:.4f}</p>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            var_cor = "#3fb950" if melhor["Var. Dia"] >= 0 else "#f85149"
            st.markdown(f"""
            <div style='background:#161b22; border:1px solid #21262d; border-radius:10px; padding:16px;'>
                <p style='color:#8b949e; font-size:11px; margin:0 0 6px;'>PREÇO / VARIAÇÃO</p>
                <p style='color:#e6edf3; font-size:20px; font-weight:800; margin:0 0 4px;'>
                    R$ {melhor["Preço"]:.2f}
                </p>
                <p style='color:{var_cor}; font-size:14px; font-weight:700; margin:0 0 4px;'>
                    {'▲' if melhor["Var. Dia"] >= 0 else '▼'} {melhor["Var. Dia"]:+.2f}%
                </p>
                <p style='color:#8b949e; font-size:12px; margin:0;'>RSI: {melhor["RSI"]:.1f}</p>
            </div>
            """, unsafe_allow_html=True)

        # Tendência macro
        bench_sma200_val = sma(prices[BENCH_TICKER], SMA200_WIN).iloc[-1] if BENCH_TICKER in prices.columns else None
        bench_price = prices[BENCH_TICKER].iloc[-1] if BENCH_TICKER in prices.columns else None
        macro_bull = bench_price > bench_sma200_val if (bench_price and bench_sma200_val and
                                                        not pd.isna(bench_sma200_val)) else True

        with col_c:
            macro_cor  = "#3fb950" if macro_bull else "#f85149"
            macro_txt  = "TENDÊNCIA ALTA ↑" if macro_bull else "TENDÊNCIA BAIXA ↓"
            macro_icon = "🟢" if macro_bull else "🔴"
            st.markdown(f"""
            <div style='background:#161b22; border:1px solid #21262d; border-radius:10px; padding:16px;'>
                <p style='color:#8b949e; font-size:11px; margin:0 0 6px;'>FILTRO MACRO (EWZ)</p>
                <p style='color:#e6edf3; font-size:20px; font-weight:800; margin:0 0 4px;'>{macro_icon} EWZ</p>
                <p style='color:{macro_cor}; font-size:14px; font-weight:700; margin:0 0 4px;'>{macro_txt}</p>
                <p style='color:#8b949e; font-size:12px; margin:0;'>vs SMA 200</p>
            </div>
            """, unsafe_allow_html=True)

        with col_d:
            compras = len(sinais_df[sinais_df["Sinal"].str.contains("COMPRA")])
            vendas  = len(sinais_df[sinais_df["Sinal"].str.contains("VENDA")])
            neutros = len(sinais_df) - compras - vendas
            st.markdown(f"""
            <div style='background:#161b22; border:1px solid #21262d; border-radius:10px; padding:16px;'>
                <p style='color:#8b949e; font-size:11px; margin:0 0 6px;'>PLACAR DE SINAIS</p>
                <p style='color:#e6edf3; font-size:14px; font-weight:700; margin:0 0 8px;'>📊 {len(sinais_df)} ativos</p>
                <p style='color:#3fb950; font-size:13px; margin:0 0 3px;'>🟢 Compra: {compras}</p>
                <p style='color:#f85149; font-size:13px; margin:0 0 3px;'>🔴 Venda: {vendas}</p>
                <p style='color:#d29922; font-size:13px; margin:0;'>🟡 Neutro: {neutros}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── CARDS DE SINAIS por ativo ────────────────────────
        st.markdown("### 🃏 Sinais Detalhados por Ativo")

        n_cols = min(len(sinais_df), 3)
        cols_cards = st.columns(n_cols)

        for idx, (_, row) in enumerate(sinais_df.iterrows()):
            col_idx = idx % n_cols
            sinal = row["Sinal"]
            cor   = row["Cor"]
            icone = row["Ícone"]
            var_cor = "#3fb950" if row["Var. Dia"] >= 0 else "#f85149"
            rsi_cor = "#f85149" if row["RSI"] > 70 else "#3fb950" if row["RSI"] < 30 else "#d29922"

            # Barra de força do sinal
            forca = row["Força"]
            barra = "".join(["█" if i < forca else "░" for i in range(5)])

            with cols_cards[col_idx]:
                st.markdown(f"""
                <div style='background:#161b22; border:1px solid {cor}44;
                            border-top: 3px solid {cor};
                            border-radius:12px; padding:18px; margin-bottom:14px;
                            transition: all 0.2s;'>
                    <div style='display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;'>
                        <div>
                            <p style='color:#e6edf3; font-size:18px; font-weight:800; margin:0;'>{icone} {row["Ticker"]}</p>
                            <p style='color:#8b949e; font-size:12px; margin:4px 0 0;'>{row["Nome"]}</p>
                        </div>
                        <span style='background:{cor}22; color:{cor}; border:1px solid {cor}55;
                                     padding:4px 10px; border-radius:20px; font-size:11px; font-weight:700;'>
                            {sinal}
                        </span>
                    </div>
                    <div style='display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:12px;'>
                        <div style='background:#0d1117; border-radius:8px; padding:10px;'>
                            <p style='color:#8b949e; font-size:10px; margin:0 0 3px;'>PREÇO</p>
                            <p style='color:#e6edf3; font-size:16px; font-weight:700; margin:0;'>R$ {row["Preço"]:.2f}</p>
                        </div>
                        <div style='background:#0d1117; border-radius:8px; padding:10px;'>
                            <p style='color:#8b949e; font-size:10px; margin:0 0 3px;'>VAR. DIA</p>
                            <p style='color:{var_cor}; font-size:16px; font-weight:700; margin:0;'>{row["Var. Dia"]:+.2f}%</p>
                        </div>
                    </div>
                    <div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:6px; margin-bottom:12px;'>
                        <div style='text-align:center;'>
                            <p style='color:#8b949e; font-size:10px; margin:0 0 2px;'>RSI</p>
                            <p style='color:{rsi_cor}; font-size:14px; font-weight:700; margin:0;'>{row["RSI"]:.1f}</p>
                        </div>
                        <div style='text-align:center;'>
                            <p style='color:#8b949e; font-size:10px; margin:0 0 2px;'>SCORE</p>
                            <p style='color:#e6edf3; font-size:14px; font-weight:700; margin:0;'>{row["Score"]:.4f}</p>
                        </div>
                        <div style='text-align:center;'>
                            <p style='color:#8b949e; font-size:10px; margin:0 0 2px;'>MACD ↑</p>
                            <p style='color:{"#3fb950" if row["MACD Bull"] else "#f85149"}; font-size:14px; font-weight:700; margin:0;'>
                                {"SIM" if row["MACD Bull"] else "NÃO"}
                            </p>
                        </div>
                    </div>
                    <div>
                        <p style='color:#8b949e; font-size:10px; margin:0 0 4px;'>FORÇA DO SINAL</p>
                        <p style='color:{cor}; font-size:16px; letter-spacing:2px; margin:0; font-family:monospace;'>{barra}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── TABELA DE SINAIS ─────────────────────────────────
        st.markdown("### 📋 Tabela Resumo de Sinais")

        tabela_exib = sinais_df[["Ticker", "Nome", "Preço", "Var. Dia", "Score",
                                  "RSI", "MACD Bull", "Acima SMA50", "Sinal"]].copy()
        tabela_exib["Preço"] = tabela_exib["Preço"].apply(lambda x: f"R$ {x:.2f}")
        tabela_exib["Var. Dia"] = tabela_exib["Var. Dia"].apply(lambda x: f"{x:+.2f}%")
        tabela_exib["Score"] = tabela_exib["Score"].apply(lambda x: f"{x:.4f}")
        tabela_exib["RSI"] = tabela_exib["RSI"].apply(lambda x: f"{x:.1f}")
        tabela_exib["MACD Bull"] = tabela_exib["MACD Bull"].apply(lambda x: "✅" if x else "❌")
        tabela_exib["Acima SMA50"] = tabela_exib["Acima SMA50"].apply(lambda x: "✅" if x else "❌")

        st.dataframe(tabela_exib, use_container_width=True, hide_index=True)

        # ── GRÁFICO DE SCORES ────────────────────────────────
        st.markdown("### 📊 Ranking de Scores — Modelo Momentum")

        cores_bar = []
        for s in sinais_df["Sinal"]:
            if "COMPRA" in s:   cores_bar.append("#3fb950")
            elif "VENDA" in s:  cores_bar.append("#f85149")
            else:               cores_bar.append("#d29922")

        fig_score = go.Figure(go.Bar(
            x=sinais_df["Ticker"],
            y=sinais_df["Score"],
            marker_color=cores_bar,
            text=[f"{v:.4f}" for v in sinais_df["Score"]],
            textposition="outside",
            textfont=dict(color="#e6edf3", size=11),
        ))
        fig_score.add_hline(y=0.02, line_dash="dash", line_color="#3fb950",
                            annotation_text="Limiar de Compra (0.02)",
                            annotation_font_color="#3fb950")
        fig_score.add_hline(y=-0.02, line_dash="dash", line_color="#f85149",
                            annotation_text="Limiar de Venda (-0.02)",
                            annotation_font_color="#f85149")
        fig_score.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b0e14",
            plot_bgcolor="#161b22",
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d", zeroline=True, zerolinecolor="#30363d"),
            showlegend=False,
        )
        st.plotly_chart(fig_score, use_container_width=True)

        # ── GRÁFICO DE RSI ───────────────────────────────────
        st.markdown("### 🎯 RSI dos Ativos — Zonas de Sobrecompra/Sobrevenda")

        fig_rsi = go.Figure()
        fig_rsi.add_hrect(y0=70, y1=100, fillcolor="#f85149", opacity=0.08,
                           annotation_text="Sobrecomprado (Venda)", annotation_position="right",
                           annotation_font_color="#f85149")
        fig_rsi.add_hrect(y0=0, y1=30, fillcolor="#3fb950", opacity=0.08,
                           annotation_text="Sobrevendido (Compra)", annotation_position="right",
                           annotation_font_color="#3fb950")
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#f85149", line_width=1)
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#3fb950", line_width=1)
        fig_rsi.add_hline(y=50, line_dash="dot", line_color="#8b949e", line_width=1)

        rsi_cores = ["#3fb950" if r < 30 else "#f85149" if r > 70 else "#d29922"
                     for r in sinais_df["RSI"]]

        fig_rsi.add_trace(go.Bar(
            x=sinais_df["Ticker"],
            y=sinais_df["RSI"],
            marker_color=rsi_cores,
            text=[f"{v:.1f}" for v in sinais_df["RSI"]],
            textposition="outside",
            textfont=dict(color="#e6edf3", size=11),
        ))
        fig_rsi.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b0e14",
            plot_bgcolor="#161b22",
            height=340,
            margin=dict(l=20, r=120, t=10, b=20),
            yaxis=dict(range=[0, 105], gridcolor="#21262d"),
            xaxis=dict(gridcolor="#21262d"),
            showlegend=False,
        )
        st.plotly_chart(fig_rsi, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 2: BACKTEST & PERFORMANCE
# ═══════════════════════════════════════════════════════════
with tab2:
    if executar or "equity" not in st.session_state:
        with st.spinner("⚙️ Rodando backtest..."):
            equity, holdings_df = run_backtest(prices, indicators, valid)
            returns = equity.pct_change().dropna()
            metricas = calcular_metricas(equity, returns)
            st.session_state["equity"]      = equity
            st.session_state["holdings_df"] = holdings_df
            st.session_state["returns"]     = returns
            st.session_state["metricas"]    = metricas
    else:
        equity      = st.session_state["equity"]
        holdings_df = st.session_state["holdings_df"]
        returns     = st.session_state["returns"]
        metricas    = st.session_state["metricas"]

    if not metricas:
        st.warning("Dados insuficientes para calcular métricas.")
        st.stop()

    # ── MÉTRICAS DE DESTAQUE ──────────────────────────────────
    st.markdown("### 📊 Métricas de Performance")
    m1, m2, m3, m4, m5 = st.columns(5)

    retorno_total = metricas.get("Retorno Total", 0)
    sharpe        = metricas.get("Índice de Sharpe", 0)
    win_rate      = metricas.get("Taxa de Vitórias", 0)
    max_dd        = metricas.get("Drawdown Máximo", 0)
    cagr          = metricas.get("CAGR", 0)

    m1.metric("📈 Retorno Total",   f"{retorno_total:.1%}", delta=f"CAGR {cagr:.1%}")
    m2.metric("⚡ Sharpe Ratio",    f"{sharpe:.2f}",  delta="↑ Bom > 1.0" if sharpe > 1 else "↓ Abaixo de 1.0")
    m3.metric("🎯 Win Rate",        f"{win_rate:.1%}", delta=f"dias pos: {int(win_rate * len(returns))}")
    m4.metric("📉 Max Drawdown",    f"{max_dd:.1%}",  delta=f"Calmar: {metricas.get('Índice de Calmar',0):.2f}")
    m5.metric("💰 Patrimônio Final", f"R$ {metricas.get('Patrimônio Final',0):,.0f}",
              delta=f"+R$ {metricas.get('Patrimônio Final',0) - CAPITAL_INICIAL:,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CURVA DE CAPITAL ─────────────────────────────────────
    st.markdown("### 📈 Curva de Capital vs Benchmark")

    fig_equity = go.Figure()

    # Estratégia
    fig_equity.add_trace(go.Scatter(
        x=equity.index, y=equity,
        name="Estratégia Momentum",
        line=dict(color="#3fb950", width=2),
        fill="tozeroy",
        fillcolor="rgba(63,185,80,0.06)",
    ))

    # Benchmark B&H
    if BENCH_TICKER in prices.columns:
        bench_idx = min(100, len(prices) - 1)
        bh = prices[BENCH_TICKER].iloc[bench_idx:] / prices[BENCH_TICKER].iloc[bench_idx] * CAPITAL_INICIAL
        fig_equity.add_trace(go.Scatter(
            x=bh.index, y=bh,
            name="EWZ Buy & Hold",
            line=dict(color="#4a90e2", width=1.5, dash="dash"),
        ))

    # Linha de capital inicial
    fig_equity.add_hline(y=CAPITAL_INICIAL, line_dash="dot",
                          line_color="#8b949e", line_width=1,
                          annotation_text=f"Capital Inicial R${CAPITAL_INICIAL/1e6:.1f}M")

    fig_equity.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0e14",
        plot_bgcolor="#161b22",
        height=420,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d",
                   tickformat="R$,.0f",
                   tickprefix="R$ "),
        legend=dict(x=0.01, y=0.99, bgcolor="#0d1117",
                    bordercolor="#21262d", borderwidth=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig_equity, use_container_width=True)

    # ── DRAWDOWN ─────────────────────────────────────────────
    col_dd, col_hist = st.columns([3, 2])

    with col_dd:
        st.markdown("### 📉 Drawdown")
        roll_max = equity.cummax()
        dd_series = (equity - roll_max) / roll_max * 100

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd_series.index, y=dd_series,
            fill="tozeroy",
            fillcolor="rgba(248,81,73,0.25)",
            line=dict(color="#f85149", width=1),
            name="Drawdown",
        ))
        fig_dd.add_hline(y=max_dd * 100, line_dash="dash",
                          line_color="#d29922", line_width=1.5,
                          annotation_text=f"Max DD: {max_dd:.1%}")
        fig_dd.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b0e14",
            plot_bgcolor="#161b22",
            height=280,
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d", ticksuffix="%"),
            showlegend=False,
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    with col_hist:
        st.markdown("### 📊 Distribuição de Retornos")
        ret_pct = returns * 100
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=ret_pct,
            nbinsx=60,
            marker_color="#4a90e2",
            opacity=0.75,
            name="Retornos Diários",
        ))
        fig_hist.add_vline(x=ret_pct.mean(), line_dash="dash",
                            line_color="#3fb950",
                            annotation_text=f"Média {ret_pct.mean():.2f}%",
                            annotation_font_color="#3fb950")
        fig_hist.add_vline(x=ret_pct.quantile(0.05), line_dash="dot",
                            line_color="#f85149",
                            annotation_text=f"VaR 5%",
                            annotation_font_color="#f85149")
        fig_hist.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b0e14",
            plot_bgcolor="#161b22",
            height=280,
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(gridcolor="#21262d", ticksuffix="%"),
            yaxis=dict(gridcolor="#21262d"),
            showlegend=False,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── RETORNOS MENSAIS (HEATMAP) ────────────────────────────
    st.markdown("### 🗓️ Heatmap de Retornos Mensais")
    try:
        monthly = equity.resample("ME").last().pct_change().dropna() * 100
        monthly_df = monthly.to_frame("ret")
        monthly_df["year"]  = monthly_df.index.year
        monthly_df["month"] = monthly_df.index.month
        pivot = monthly_df.pivot(index="year", columns="month", values="ret")
        meses = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]
        pivot.columns = [meses[i-1] for i in pivot.columns]

        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[[0.0,"#7d2020"],[0.3,"#f85149"],[0.5,"#21262d"],
                        [0.7,"#2ea043"],[1.0,"#3fb950"]],
            zmid=0,
            text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row]
                  for row in pivot.values],
            texttemplate="%{text}",
            textfont=dict(size=11, color="white"),
            hovertemplate="Mês: %{x}<br>Ano: %{y}<br>Retorno: %{z:.2f}%<extra></extra>",
        ))
        fig_heat.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b0e14",
            plot_bgcolor="#161b22",
            height=300,
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(side="top"),
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    except Exception:
        st.info("Heatmap indisponível para o período selecionado.")

    # ── EXPOSIÇÃO / ALOCAÇÃO ──────────────────────────────────
    col_pie, col_metrics = st.columns([2, 3])

    with col_pie:
        st.markdown("### 🥧 Exposição por Ativo")
        hc = holdings_df["holding"].value_counts()
        cores_pie = ["#3fb950","#4a90e2","#d29922","#f85149","#9b59b6",
                     "#3fb950","#00cec9","#8b949e"]
        fig_pie = go.Figure(go.Pie(
            labels=hc.index,
            values=hc.values,
            hole=0.5,
            marker=dict(colors=cores_pie[:len(hc)],
                        line=dict(color="#0b0e14", width=2)),
            textinfo="label+percent",
            textfont=dict(size=12, color="#e6edf3"),
        ))
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b0e14",
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_metrics:
        st.markdown("### 📋 Tabela de Métricas Completa")
        rows = [
            ("💰 Patrimônio Final",       f"R$ {metricas.get('Patrimônio Final',0):,.0f}"),
            ("📈 Retorno Total",          f"{metricas.get('Retorno Total',0):.2%}"),
            ("🚀 CAGR",                   f"{metricas.get('CAGR',0):.2%}"),
            ("⚡ Índice de Sharpe",       f"{metricas.get('Índice de Sharpe',0):.3f}"),
            ("🎯 Sortino",                f"{metricas.get('Proporção de Sortino',0):.3f}"),
            ("📉 Drawdown Máximo",        f"{metricas.get('Drawdown Máximo',0):.2%}"),
            ("🏆 Índice de Calmar",       f"{metricas.get('Índice de Calmar',0):.3f}"),
            ("📊 Volatilidade Anual",     f"{metricas.get('Volatilidade Anual',0):.2%}"),
            ("✅ Taxa de Vitórias",       f"{metricas.get('Taxa de Vitórias',0):.1%}"),
            ("💚 Vitória Média",          f"{metricas.get('Vitória Média',0):.2%}"),
            ("🔴 Perda Média",            f"{metricas.get('Perda Média',0):.2%}"),
            ("⚖️ Lucro/Prejuízo",        f"{metricas.get('Índice Lucro/Prejuízo',0):.2f}"),
        ]
        df_met = pd.DataFrame(rows, columns=["Métrica", "Valor"])
        st.dataframe(df_met, use_container_width=True, hide_index=True, height=380)


# ═══════════════════════════════════════════════════════════
# TAB 3: ANÁLISE TÉCNICA POR ATIVO
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🔬 Análise Técnica Detalhada por Ativo")

    ativo_sel = st.selectbox(
        "Selecione o ativo para análise:",
        valid,
        format_func=lambda x: f"{x} — {TICKERS_DEFAULT.get(x, x)}"
    )

    if ativo_sel and ativo_sel in indicators:
        ind = indicators[ativo_sel]
        s   = prices[ativo_sel].dropna()
        n   = min(252, len(s))
        s_plot = s.iloc[-n:]

        # ── PREÇO + BOLLINGER + SMAs ──────────────────────────
        st.markdown(f"#### 📊 {ativo_sel} — Preço, Médias e Bollinger Bands")
        fig_price = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.55, 0.25, 0.20],
            vertical_spacing=0.04,
            subplot_titles=["Preço + SMAs + Bollinger", "MACD", "RSI"]
        )

        # Candlestick-like com linha
        fig_price.add_trace(go.Scatter(
            x=s_plot.index, y=s_plot.values,
            name="Preço",
            line=dict(color="#e6edf3", width=1.5),
        ), row=1, col=1)

        # Bollinger Bands
        if not pd.isna(ind["bb_up"].iloc[-1]):
            bb_up_p   = ind["bb_up"].iloc[-n:]
            bb_mid_p  = ind["bb_mid"].iloc[-n:]
            bb_low_p  = ind["bb_low"].iloc[-n:]
            fig_price.add_trace(go.Scatter(
                x=bb_up_p.index, y=bb_up_p.values, name="BB Superior",
                line=dict(color="#4a90e2", width=1, dash="dot"), opacity=0.6,
            ), row=1, col=1)
            fig_price.add_trace(go.Scatter(
                x=bb_low_p.index, y=bb_low_p.values, name="BB Inferior",
                line=dict(color="#4a90e2", width=1, dash="dot"), opacity=0.6,
                fill="tonexty", fillcolor="rgba(74,144,226,0.06)",
            ), row=1, col=1)
            fig_price.add_trace(go.Scatter(
                x=bb_mid_p.index, y=bb_mid_p.values, name="BB Média",
                line=dict(color="#4a90e2", width=0.8), opacity=0.4,
            ), row=1, col=1)

        # SMA 50
        sma50_p = ind["sma50"].iloc[-n:]
        fig_price.add_trace(go.Scatter(
            x=sma50_p.index, y=sma50_p.values, name="SMA 50",
            line=dict(color="#d29922", width=1.2),
        ), row=1, col=1)

        # SMA 200
        sma200_p = ind["sma200"].iloc[-n:]
        fig_price.add_trace(go.Scatter(
            x=sma200_p.index, y=sma200_p.values, name="SMA 200",
            line=dict(color="#f85149", width=1.2, dash="dash"),
        ), row=1, col=1)

        # MACD
        macd_p = ind["macd"].iloc[-n:]
        sig_p  = ind["macd_sig"].iloc[-n:]
        hist_p = ind["macd_hist"].iloc[-n:]
        hist_colors = ["#3fb950" if v >= 0 else "#f85149" for v in hist_p.fillna(0)]

        fig_price.add_trace(go.Bar(
            x=hist_p.index, y=hist_p.values, name="Histograma",
            marker_color=hist_colors, opacity=0.7,
        ), row=2, col=1)
        fig_price.add_trace(go.Scatter(
            x=macd_p.index, y=macd_p.values, name="MACD",
            line=dict(color="#4a90e2", width=1.5),
        ), row=2, col=1)
        fig_price.add_trace(go.Scatter(
            x=sig_p.index, y=sig_p.values, name="Sinal",
            line=dict(color="#d29922", width=1.5),
        ), row=2, col=1)

        # RSI
        rsi_p = ind["rsi"].iloc[-n:]
        rsi_colors_line = []
        for v in rsi_p.fillna(50):
            if v > 70:   rsi_colors_line.append("#f85149")
            elif v < 30: rsi_colors_line.append("#3fb950")
            else:        rsi_colors_line.append("#d29922")

        fig_price.add_trace(go.Scatter(
            x=rsi_p.index, y=rsi_p.values, name="RSI",
            line=dict(color="#d29922", width=1.5),
            fill="tozeroy", fillcolor="rgba(210,153,34,0.06)",
        ), row=3, col=1)
        fig_price.add_hline(y=70, line_dash="dash", line_color="#f85149",
                             line_width=1, row=3, col=1)
        fig_price.add_hline(y=30, line_dash="dash", line_color="#3fb950",
                             line_width=1, row=3, col=1)
        fig_price.add_hline(y=50, line_dash="dot", line_color="#8b949e",
                             line_width=0.8, row=3, col=1)

        # Anotação de sinal no gráfico
        sinal_ativo = sinais_df[sinais_df["Ticker"] == ativo_sel]
        if not sinal_ativo.empty:
            sa = sinal_ativo.iloc[0]
            sinal_cor = "#3fb950" if "COMPRA" in sa["Sinal"] else "#f85149" if "VENDA" in sa["Sinal"] else "#d29922"
            fig_price.add_annotation(
                x=s_plot.index[-1], y=s_plot.values[-1],
                text=f"  {sa['Ícone']} {sa['Sinal']}",
                showarrow=True, arrowhead=2,
                arrowcolor=sinal_cor,
                font=dict(color=sinal_cor, size=13, family="monospace"),
                bgcolor="#161b22",
                bordercolor=sinal_cor,
                borderwidth=1,
                row=1, col=1
            )

        fig_price.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b0e14",
            plot_bgcolor="#161b22",
            height=680,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(bgcolor="#0d1117", bordercolor="#21262d", borderwidth=1,
                        x=0.01, y=0.99, font=dict(size=11)),
            hovermode="x unified",
        )
        for row in [1, 2, 3]:
            fig_price.update_xaxes(gridcolor="#21262d", row=row, col=1)
            fig_price.update_yaxes(gridcolor="#21262d", row=row, col=1)
        fig_price.update_yaxes(ticksuffix="", row=3, col=1, range=[0, 100])

        st.plotly_chart(fig_price, use_container_width=True)

        # ── MINI MÉTRICAS DO ATIVO ────────────────────────────
        st.markdown(f"#### 📋 Métricas Rápidas — {ativo_sel}")
        c1, c2, c3, c4, c5 = st.columns(5)
        p = prices[ativo_sel].dropna()
        ret_30d = (p.iloc[-1] / p.iloc[-min(21, len(p))] - 1) * 100
        ret_63d = (p.iloc[-1] / p.iloc[-min(63, len(p))] - 1) * 100
        vol_30d = p.pct_change().rolling(21).std().iloc[-1] * np.sqrt(252) * 100
        rsi_now = ind["rsi"].iloc[-1]
        macd_now= ind["macd"].iloc[-1]
        sig_now = ind["macd_sig"].iloc[-1]

        c1.metric("Preço Atual",   f"R$ {p.iloc[-1]:.2f}")
        c2.metric("Ret. 1M",       f"{ret_30d:+.1f}%")
        c3.metric("Ret. 3M",       f"{ret_63d:+.1f}%")
        c4.metric("RSI (14)",      f"{rsi_now:.1f}",
                  delta="Sobrecomprado" if rsi_now > 70 else "Sobrevendido" if rsi_now < 30 else "Normal")
        c5.metric("Vol. Anual",    f"{vol_30d:.1f}%")

# ── Rodapé ──────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#8b949e; font-size:12px; padding:10px 0;'>
    📊 <b>Momentum Rotacional B3</b> — Dashboard Quant v2.0 &nbsp;|&nbsp;
    Dados: Yahoo Finance &nbsp;|&nbsp; Atualização automática a cada 60s &nbsp;|&nbsp;
    <span style='color:#f85149;'>⚠️ Este dashboard é apenas para fins educacionais. Não é recomendação de investimento.</span>
</div>
""", unsafe_allow_html=True)
