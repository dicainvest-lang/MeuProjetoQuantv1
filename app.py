# -*- coding: utf-8 -*-
# ==============================================================================
# ROBÔ IA OURO v7.0 — STREAMLIT DASHBOARD EDITION
# Fusão: bot_ouro_mt5_corrigido H4 v2 + Streamlit UI (Dark Mode)
# Ativo padrão: XAU/USD (GC=F) via yfinance | Timeframe: 1h (proxy H4)
# ==============================================================================

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json, os

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_OK = True
except ImportError:
    AUTOREFRESH_OK = False

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Robô IA Ouro v7.0",
    page_icon="🥇",
    layout="wide",
)

st.markdown("""
<style>
    :root {
        --bg-dark:#0b0e14; --bg-card:#131720; --border:#2a3048;
        --text-main:#e8eaf6; --text-sub:#8b949e;
        --gold:#c9a84c; --gold-lt:#f0d080;
        --success:#3fb950; --danger:#f85149;
        --warning:#d29922; --blue:#58a6ff;
    }
    .stApp { background-color: var(--bg-dark); color: var(--text-main); }
    section[data-testid="stSidebar"] {
        background-color: var(--bg-card);
        border-right: 1px solid var(--border);
    }
    .metric-card {
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: 12px; padding: 18px 20px; text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.5);
    }
    .metric-card .label { color:var(--text-sub); font-size:0.75rem;
        text-transform:uppercase; letter-spacing:1px; margin-bottom:6px; }
    .metric-card .value { font-size:1.8rem; font-weight:700; }
    .metric-card .sub   { font-size:0.72rem; color:var(--text-sub); margin-top:4px; }
    .sinal-box {
        border-radius: 16px; padding: 28px; text-align: center; margin: 16px 0;
        border: 2px solid;
    }
    .gold { color: #c9a84c; } .green { color: #3fb950; }
    .red  { color: #f85149; } .blue  { color: #58a6ff; }
    .yellow { color: #d29922; }
    h1,h2,h3 { color: var(--text-main) !important; }
    .stButton>button {
        background: linear-gradient(135deg, #c9a84c, #f0d080);
        color: #0b0e14; border: none; border-radius: 8px;
        font-weight: 800; padding: 0.6rem 1.8rem; font-size: 1rem;
        width: 100%;
    }
    .stButton>button:hover { opacity: 0.88; }
    div[data-testid="stMetric"] {
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: 10px; padding: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("## 🥇 Robô IA Ouro v7.0")
    st.markdown("---")

    TICKER = st.text_input(
        "🔎 Ativo",
        value="GC=F",
        help="GC=F = Ouro Futuro | XAUUSD=X = Spot | GLD = ETF Ouro"
    )
    periodo_map = {"1 Ano":1, "2 Anos":2, "3 Anos":3, "5 Anos":5}
    periodo_label = st.selectbox("📅 Período de Histórico", list(periodo_map.keys()), index=2)
    ANOS = periodo_map[periodo_label]

    TIMEFRAME = st.selectbox("⏱️ Timeframe", ["1h","1d"], index=0,
                              help="1h ≈ H4 do MT5 (4 candles). 1d = diário.")

    st.markdown("### ⚙️ Parâmetros da Estratégia")
    PROB_THRESHOLD  = st.slider("Threshold de Probabilidade", 0.50, 0.90, 0.55, 0.01)
    Z_THRESHOLD     = st.slider("Z-Score Threshold", 0.5, 3.0, 1.0, 0.1)
    ATR_MULT_SL     = st.slider("ATR Mult. Stop Loss",    0.3, 2.0, 0.5, 0.1)
    ATR_MULT_TP     = st.slider("ATR Mult. Take Profit",  0.5, 5.0, 1.5, 0.1)
    HORIZONTE       = st.slider("Horizonte (candles)", 1, 20, 5, 1,
                                help="Quantos candles à frente prever (5 H4 ≈ 20h)")
    CAPITAL         = st.number_input("Capital (USD)", value=10_000, step=1_000)
    RISCO_TRADE_PCT = st.slider("Risco por Trade (%)", 0.5, 5.0, 2.0, 0.5) / 100
    MAX_DD_PCT      = st.slider("Max Drawdown Diário (%)", 1.0, 10.0, 5.0, 0.5) / 100

    st.markdown("### 🔄 Auto-Refresh")
    refresh_sel = st.selectbox("Intervalo", ["Desligado","1 min","5 min"], index=0)

    run_btn = st.button("🚀 Executar Análise")

if AUTOREFRESH_OK and refresh_sel != "Desligado":
    ms = 60_000 if refresh_sel == "1 min" else 300_000
    st_autorefresh(interval=ms, key="autorefresh")

# Título
st.markdown("# 🥇 Robô IA Ouro v7.0 — Dashboard")
st.markdown(f"**Ativo:** `{TICKER.upper()}` &nbsp;|&nbsp; **Timeframe:** `{TIMEFRAME}` "
            f"&nbsp;|&nbsp; **Período:** {periodo_label} "
            f"&nbsp;|&nbsp; **Atualizado:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
st.markdown("---")

# ==============================================================================
# FUNÇÕES CORE — lógica idêntica ao bot_ouro_mt5_corrigido H4 v2
# ==============================================================================

@st.cache_data(ttl=60)
def baixar_dados(ticker: str, anos: int, interval: str) -> pd.DataFrame:
    start = (datetime.now() - timedelta(days=anos * 365)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, interval=interval,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().capitalize() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    for alt in ['Adj close','Adj_close','Adjclose']:
        if alt in df.columns and 'Close' not in df.columns:
            df.rename(columns={alt:'Close'}, inplace=True)
    if 'Volume' not in df.columns:
        df['Volume'] = 0.0
    df['Volume'] = df['Volume'].fillna(0).astype(float)
    for col in ['Open','High','Low','Close']:
        if col in df.columns:
            df[col] = df[col].astype(float)
    df.dropna(subset=['Open','High','Low','Close'], inplace=True)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='last')]
    return df


def test_stationarity(series: pd.Series) -> bool:
    """Teste ADF: retorna True se a série for estacionária (p < 0.05)."""
    try:
        return adfuller(series.dropna())[1] < 0.05
    except:
        return False


def calcular_rsi(retornos: pd.Series, periodo: int = 14) -> pd.Series:
    ganhos = retornos.clip(lower=0).rolling(periodo).mean()
    perdas = (-retornos.clip(upper=0)).rolling(periodo).mean()
    rs = ganhos / perdas.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def preparar_features(df_raw: pd.DataFrame, horizonte: int = 5) -> pd.DataFrame:
    """
    Replica exata da lógica do bot_ouro_mt5_corrigido H4 v2.
    Features: RSI, Z_Score, Volatilidade, Is_Stationary, ATR_Pct
    Target: 1=COMPRA | -1=VENDA | 0=NEUTRO (baseado em percentil 25)
    """
    df = df_raw.copy()
    df.columns = [c.capitalize() for c in df.columns]

    # Renomeia Close → Ouro (mantém compatibilidade com o bot original)
    if 'Close' in df.columns:
        df.rename(columns={'Close':'Ouro'}, inplace=True)
    if 'Ouro' not in df.columns:
        raise ValueError("Coluna 'Close' não encontrada nos dados.")

    df['Retorno'] = df['Ouro'].pct_change()

    # Z-Score sobre PREÇOS (mean reversion) — igual ao bot original
    df['Z_Score'] = df['Ouro'].rolling(100).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0.0,
        raw=False
    )

    # Estacionariedade sobre RETORNOS — rolling ADF
    df['Is_Stationary'] = df['Retorno'].rolling(100).apply(
        lambda x: 1.0 if test_stationarity(pd.Series(x)) else 0.0,
        raw=False
    )

    df['RSI']          = calcular_rsi(df['Retorno'])
    df['ATR']          = (df['High'] - df['Low']).rolling(14).mean()
    df['ATR_Pct']      = df['ATR'] / df['Ouro']
    df['Volatilidade'] = df['Retorno'].rolling(20).std()

    # Target: retorno futuro a N candles
    ret_fut = df['Ouro'].pct_change(horizonte).shift(-horizonte)
    limiar  = np.percentile(np.abs(df['Retorno'].dropna()), 25)
    df['Alvo']       = np.select([ret_fut > limiar, ret_fut < -limiar], [1, -1], default=0)
    df['Ret_Futuro'] = ret_fut

    # Features extras para visualização
    df['Ouro_Close']  = df['Ouro']
    df['SMA_20']      = df['Ouro'].rolling(20).mean()
    df['SMA_50']      = df['Ouro'].rolling(50).mean()
    df['BB_Upper']    = df['SMA_20'] + 2 * df['Ouro'].rolling(20).std()
    df['BB_Lower']    = df['SMA_20'] - 2 * df['Ouro'].rolling(20).std()

    # MACD
    ema12 = df['Ouro'].ewm(span=12, adjust=False).mean()
    ema26 = df['Ouro'].ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']

    return df.dropna()


FEATURES = ['RSI', 'Z_Score', 'Volatilidade', 'Is_Stationary', 'ATR_Pct']


def treinar_modelo(df_feat: pd.DataFrame):
    """Random Forest — exatamente como no bot original (80/20 split temporal)."""
    X = df_feat[FEATURES]
    y = df_feat['Alvo']
    split = int(len(X) * 0.80)
    X_tr, X_val = X.iloc[:split], X.iloc[split:]
    y_tr, y_val = y.iloc[:split], y.iloc[split:]

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_tr, y_tr)

    y_pred_val = rf.predict(X_val)
    acc  = accuracy_score(y_val, y_pred_val)
    prec = precision_score(y_val, y_pred_val, average='weighted', zero_division=0)
    rec  = recall_score(y_val, y_pred_val, average='weighted', zero_division=0)
    f1   = f1_score(y_val, y_pred_val, average='weighted', zero_division=0)

    return rf, {'acc':acc, 'prec':prec, 'rec':rec, 'f1':f1,
                'n_train':split, 'n_val':len(X_val)}


def calcular_sinal(modelo, df_feat, prob_threshold, z_threshold) -> dict:
    """
    Replica a lógica de sinal do bot:
    Compra: prob_compra > threshold AND z < -z_threshold AND estacionária AND atr > 0
    Venda : prob_venda  > threshold AND z > +z_threshold AND estacionária AND atr > 0
    """
    ult = df_feat.iloc[-1]
    classes = list(modelo.classes_)
    X_ult   = pd.DataFrame([{f: ult[f] for f in FEATURES}])
    probs   = modelo.predict_proba(X_ult)[0]

    idx_c = classes.index(1)  if  1 in classes else None
    idx_v = classes.index(-1) if -1 in classes else None

    p_compra = float(probs[idx_c]) if idx_c is not None else 0.0
    p_venda  = float(probs[idx_v]) if idx_v is not None else 0.0
    p_neutro = 1.0 - p_compra - p_venda

    z    = float(ult['Z_Score'])
    est  = bool(ult['Is_Stationary'] == 1.0)
    atr  = float(ult['ATR_Pct'])
    rsi  = float(ult['RSI'])
    vol  = float(ult['Volatilidade'])

    conds = {
        f"Prob. Compra {p_compra*100:.1f}% > {prob_threshold*100:.0f}%": p_compra > prob_threshold,
        f"Prob. Venda {p_venda*100:.1f}% > {prob_threshold*100:.0f}%":   p_venda  > prob_threshold,
        f"Z-Score {z:+.3f} (lim ±{z_threshold})":                        abs(z) > z_threshold,
        "Série Estacionária":                                              est,
        f"ATR > 0 ({atr:.4f})":                                           atr > 0,
    }

    direcao = 0
    if p_compra > prob_threshold and z < -z_threshold and est and atr > 0:
        direcao =  1
    elif p_venda > prob_threshold and z >  z_threshold and est and atr > 0:
        direcao = -1

    sinal_str = "COMPRA 🟢" if direcao == 1 else "VENDA 🔴" if direcao == -1 else "NEUTRO ⚪"
    confianca = p_compra if direcao == 1 else p_venda if direcao == -1 else max(p_compra, p_venda)

    return {
        'direcao':  direcao,
        'sinal':    sinal_str,
        'confianca':confianca,
        'p_compra': p_compra,
        'p_venda':  p_venda,
        'p_neutro': p_neutro,
        'z':        z,
        'est':      est,
        'atr_pct':  atr,
        'rsi':      rsi,
        'vol':      vol * 100,
        'conds':    conds,
        'preco':    float(ult['Ouro']),
    }


def calcular_lote_sl_tp(preco, atr_pct, capital, risco_pct, mult_sl, mult_tp):
    """Replica o cálculo de lote/SL/TP do bot (simplificado para yfinance)."""
    sl_dist = atr_pct * mult_sl
    tp_dist = atr_pct * mult_tp
    risco_usd = capital * risco_pct
    # Para GC=F: 1 contrato = 100 oz ≈ equiv
    sl_usd_por_lote = preco * sl_dist * 100
    lote = (risco_usd / sl_usd_por_lote) if sl_usd_por_lote > 0 else 0.0
    lote = max(0.01, min(lote, 50.0))
    sl_usd = round(risco_usd, 2)
    tp_usd = round(risco_usd * (mult_tp / mult_sl), 2)
    rr     = round(mult_tp / mult_sl, 2)
    return {
        'lote':   round(lote, 2),
        'sl_pct': sl_dist,
        'tp_pct': tp_dist,
        'sl_usd': sl_usd,
        'tp_usd': tp_usd,
        'rr':     rr,
        'risco_capital_pct': risco_pct * 100,
    }


def backtest_simples(df_feat, modelo, prob_threshold, z_threshold,
                      capital, risco_pct, mult_sl, mult_tp, custo=0.0005):
    """Backtest candle-a-candle replicando a lógica de sinal do bot."""
    classes = list(modelo.classes_)
    X       = df_feat[FEATURES]
    probs   = modelo.predict_proba(X)
    idx_c   = classes.index(1)  if  1 in classes else 0
    idx_v   = classes.index(-1) if -1 in classes else 0

    cap     = capital
    equity  = [cap]
    trades  = []
    precos  = df_feat['Ouro'].values
    rets    = df_feat['Ret_Futuro'].values
    zs      = df_feat['Z_Score'].values
    ests    = df_feat['Is_Stationary'].values
    atrs    = df_feat['ATR_Pct'].values
    dates   = df_feat.index

    for i in range(len(df_feat)):
        p_c = probs[i][idx_c]
        p_v = probs[i][idx_v]
        z   = zs[i]
        est = ests[i] == 1.0
        atr = atrs[i]
        ret = rets[i] if not np.isnan(rets[i]) else 0.0

        d = 0
        if p_c > prob_threshold and z < -z_threshold and est and atr > 0: d =  1
        elif p_v > prob_threshold and z >  z_threshold and est and atr > 0: d = -1

        if d == 0:
            equity.append(cap)
            continue

        sl_d = atr * mult_sl
        tp_d = atr * mult_tp
        risco_usd = cap * risco_pct
        tamanho   = risco_usd

        pnl = tamanho * d * ret
        if d == 1:
            if ret < -sl_d: pnl = -tamanho * sl_d
            elif ret >  tp_d: pnl =  tamanho * tp_d
        else:
            if ret >  sl_d: pnl = -tamanho * sl_d
            elif ret < -tp_d: pnl =  tamanho * tp_d

        pnl -= cap * custo
        cap += pnl
        equity.append(cap)
        trades.append({'date':dates[i], 'dir':d, 'ret':ret, 'pnl':pnl})

    equity = np.array(equity)
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / peak
    rets_eq = np.diff(equity) / equity[:-1]

    n_t   = len(trades)
    win_r = np.mean([t['pnl']>0 for t in trades])*100 if n_t else 0.0
    sharpe= float(rets_eq.mean()/rets_eq.std()*np.sqrt(252)) if rets_eq.std()>0 else 0.0
    ret_a = (equity[-1]/capital-1)*100

    # BH Ouro
    bh = (precos[-1]/precos[0]-1)*100 if precos[0]>0 else 0.0

    return {
        'equity':equity, 'dd':dd, 'trades':trades,
        'ret_acum':ret_a, 'bh':bh, 'sharpe':sharpe,
        'max_dd':float(dd.min()*100), 'win_rate':win_r,
        'n_trades':n_t, 'cap_final':cap,
        'dates':df_feat.index[:len(equity)],
    }

# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================
if run_btn or 'results' not in st.session_state:
    with st.spinner(f"⏳ Baixando `{TICKER}` e treinando Random Forest..."):
        try:
            df_raw = baixar_dados(TICKER.upper(), ANOS, TIMEFRAME)
            if len(df_raw) < 200:
                st.error("⚠️ Dados insuficientes. Use período maior ou troque o ativo.")
                st.stop()
            df_feat = preparar_features(df_raw, horizonte=HORIZONTE)
            if len(df_feat) < 120:
                st.error(f"⚠️ Após feature engineering: {len(df_feat)} linhas. Aumente o período.")
                st.stop()
            modelo, metricas_modelo = treinar_modelo(df_feat)
            sinal = calcular_sinal(modelo, df_feat, PROB_THRESHOLD, Z_THRESHOLD)
            lote_info = calcular_lote_sl_tp(
                sinal['preco'], sinal['atr_pct'],
                CAPITAL, RISCO_TRADE_PCT, ATR_MULT_SL, ATR_MULT_TP
            )
            bt = backtest_simples(
                df_feat, modelo, PROB_THRESHOLD, Z_THRESHOLD,
                CAPITAL, RISCO_TRADE_PCT, ATR_MULT_SL, ATR_MULT_TP
            )
            st.session_state['results'] = {
                'df_raw': df_raw, 'df_feat': df_feat,
                'modelo': modelo, 'metricas': metricas_modelo,
                'sinal': sinal, 'lote_info': lote_info, 'bt': bt,
            }
        except Exception as e:
            st.error(f"Erro: {e}")
            st.stop()

res = st.session_state.get('results')
if not res:
    st.info("Configure os parâmetros e clique em **🚀 Executar Análise**.")
    st.stop()

s   = res['sinal']
bt  = res['bt']
li  = res['lote_info']
met = res['metricas']

# ==============================================================================
# CARDS DE MÉTRICAS
# ==============================================================================
st.markdown("## 📊 Resumo da Estratégia")

def card(col, label, value, css, sub=""):
    col.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value {css}">{value}</div>
        <div class="sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)
card(c1, "Retorno Acumulado", f"{bt['ret_acum']:+.2f}%",
     "green" if bt['ret_acum']>=0 else "red", f"B&H Ouro: {bt['bh']:+.2f}%")
card(c2, "Sharpe Ratio", f"{bt['sharpe']:.2f}",
     "green" if bt['sharpe']>=1 else ("blue" if bt['sharpe']>=0 else "red"), "Anualizado")
card(c3, "Win Rate", f"{bt['win_rate']:.1f}%",
     "green" if bt['win_rate']>=50 else "red", f"{bt['n_trades']} trades")
card(c4, "Max Drawdown", f"{bt['max_dd']:.2f}%", "red", "Pior queda")

st.markdown("<br>", unsafe_allow_html=True)
m1,m2,m3,m4,m5,m6 = st.columns(6)
m1.metric("Acurácia Modelo", f"{met['acc']:.1%}")
m2.metric("F1-Score",        f"{met['f1']:.3f}")
m3.metric("Capital Final",   f"${bt['cap_final']:,.0f}")
m4.metric("Prob. Compra",    f"{s['p_compra']:.1%}")
m5.metric("Prob. Venda",     f"{s['p_venda']:.1%}")
m6.metric("Z-Score",         f"{s['z']:+.3f}")

st.markdown("---")

# ==============================================================================
# TABS
# ==============================================================================
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "🎯 Sinal Atual","📉 Backtest & Equity","📊 Indicadores",
    "✅ Checklist","🧠 Modelo"
])

PLT_BG   = '#0b0e14'
PLT_CARD = '#131720'
PLT_GRID = '#2a3048'
PLT_TEXT = '#e8eaf6'

def base_layout(h=480):
    return dict(height=h, paper_bgcolor=PLT_BG, plot_bgcolor=PLT_CARD,
                font=dict(color=PLT_TEXT), legend=dict(bgcolor=PLT_CARD),
                margin=dict(l=50,r=20,t=50,b=40))

def upd(fig):
    fig.update_xaxes(gridcolor=PLT_GRID, zeroline=False)
    fig.update_yaxes(gridcolor=PLT_GRID, zeroline=False)
    return fig

# ── Tab 1: Sinal Atual ────────────────────────────────────────────────────────
with tab1:
    sinal_color = ("#3fb950" if "COMPRA" in s['sinal'] else
                   "#f85149" if "VENDA"  in s['sinal'] else "#8b949e")

    st.markdown(f"""
    <div class="sinal-box" style="border-color:{sinal_color}; background:#131720;">
        <div style="font-size:3.2rem;margin-bottom:6px;">{s['sinal']}</div>
        <div style="color:#8b949e;font-size:0.85rem;">
            Random Forest | Z-Score: {s['z']:+.3f} | Estacionária: {'✅' if s['est'] else '❌'}
        </div>
        <div style="margin-top:22px;display:flex;justify-content:center;gap:40px;">
            <div>
                <div style="color:#8b949e;font-size:0.72rem">PROB. COMPRA</div>
                <div style="font-size:1.6rem;color:#3fb950;font-weight:700">{s['p_compra']:.1%}</div>
            </div>
            <div>
                <div style="color:#8b949e;font-size:0.72rem">PROB. VENDA</div>
                <div style="font-size:1.6rem;color:#f85149;font-weight:700">{s['p_venda']:.1%}</div>
            </div>
            <div>
                <div style="color:#8b949e;font-size:0.72rem">NEUTRO</div>
                <div style="font-size:1.6rem;color:#8b949e;font-weight:700">{s['p_neutro']:.1%}</div>
            </div>
            <div>
                <div style="color:#8b949e;font-size:0.72rem">CONFIANÇA</div>
                <div style="font-size:1.6rem;color:#c9a84c;font-weight:700">{s['confianca']:.1%}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        # Gauge probabilidades
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=s['p_compra']*100,
            number={'suffix':'%','font':{'color':PLT_TEXT}},
            gauge={
                'axis':{'range':[0,100],'tickcolor':PLT_TEXT},
                'bar':{'color':'#3fb950','thickness':0.35},
                'bgcolor':PLT_CARD,
                'steps':[{'range':[0,PROB_THRESHOLD*100],'color':'#1a0f0f'},
                         {'range':[PROB_THRESHOLD*100,100],'color':'#0f1a0f'}],
                'threshold':{'line':{'color':'#c9a84c','width':3},
                             'thickness':0.8,'value':PROB_THRESHOLD*100},
            },
            title={'text':'Prob. COMPRA (%)','font':{'color':PLT_TEXT}}
        ))
        fig_g.update_layout(paper_bgcolor=PLT_BG, font=dict(color=PLT_TEXT), height=280)
        st.plotly_chart(fig_g, use_container_width=True)

    with col_s2:
        st.markdown("#### 📐 Gestão de Risco")
        preco = s['preco']
        if s['direcao'] == 1:
            sl_p = preco * (1 - li['sl_pct'])
            tp_p = preco * (1 + li['tp_pct'])
        elif s['direcao'] == -1:
            sl_p = preco * (1 + li['sl_pct'])
            tp_p = preco * (1 - li['tp_pct'])
        else:
            sl_p = tp_p = preco

        st.markdown(f"""
        | Parâmetro | Valor |
        |:--|--:|
        | Preço Atual | `${preco:,.2f}` |
        | Stop Loss (preço) | `${sl_p:,.2f}` |
        | Take Profit (preço) | `${tp_p:,.2f}` |
        | SL em USD | `${li['sl_usd']:,.2f}` |
        | TP em USD | `${li['tp_usd']:,.2f}` |
        | R/R | `1:{li['rr']:.1f}` |
        | Risco/Capital | `{li['risco_capital_pct']:.1f}%` |
        | RSI | `{s['rsi']:.2f}` |
        | ATR % | `{s['atr_pct']*100:.4f}%` |
        | Volatilidade | `{s['vol']:.4f}%` |
        """)

    st.warning("⚠️ Sinal educacional. Não constitui recomendação de investimento.")

# ── Tab 2: Backtest & Equity ──────────────────────────────────────────────────
with tab2:
    df_f = res['df_feat']
    eq   = bt['equity']
    dd   = bt['dd']
    d_eq = bt['dates']

    fig_bt = make_subplots(rows=3, cols=1, shared_xaxes=True,
                           row_heights=[0.45,0.3,0.25],
                           subplot_titles=["Preço do Ouro + Médias Móveis",
                                           "Equity Curve (Backtest)",
                                           "Drawdown (%)"])

    # Preço + Bollinger
    fig_bt.add_trace(go.Scatter(x=df_f.index, y=df_f['Ouro'],
                                name="Ouro", line=dict(color='#c9a84c',width=1.5)), row=1,col=1)
    fig_bt.add_trace(go.Scatter(x=df_f.index, y=df_f['SMA_20'],
                                name="SMA 20", line=dict(color='#58a6ff',width=1,dash='dot')), row=1,col=1)
    fig_bt.add_trace(go.Scatter(x=df_f.index, y=df_f['SMA_50'],
                                name="SMA 50", line=dict(color='#bc8cff',width=1,dash='dot')), row=1,col=1)
    fig_bt.add_trace(go.Scatter(x=df_f.index, y=df_f['BB_Upper'],
                                name="BB +2σ", line=dict(color='#2a3048',width=0.8),
                                showlegend=False), row=1,col=1)
    fig_bt.add_trace(go.Scatter(x=df_f.index, y=df_f['BB_Lower'],
                                name="BB -2σ", line=dict(color='#2a3048',width=0.8),
                                fill='tonexty', fillcolor='rgba(42,48,72,0.2)',
                                showlegend=False), row=1,col=1)

    # Equity
    min_len = min(len(d_eq), len(eq))
    fig_bt.add_trace(go.Scatter(x=d_eq[:min_len], y=eq[:min_len],
                                name="Equity", fill='tozeroy',
                                line=dict(color='#3fb950',width=2),
                                fillcolor='rgba(63,185,80,0.1)'), row=2,col=1)
    fig_bt.add_hline(y=CAPITAL, line_dash="dash", line_color="#8b949e", row=2,col=1)

    # Drawdown
    fig_bt.add_trace(go.Scatter(x=d_eq[:min_len], y=dd[:min_len]*100,
                                name="Drawdown", fill='tozeroy',
                                line=dict(color='#f85149',width=1),
                                fillcolor='rgba(248,81,73,0.12)'), row=3,col=1)

    fig_bt.update_layout(**base_layout(640))
    upd(fig_bt)
    st.plotly_chart(fig_bt, use_container_width=True)

    # Trades tabela
    if bt['trades']:
        df_tr = pd.DataFrame(bt['trades'])
        df_tr['Direção'] = df_tr['dir'].map({1:'COMPRA 🟢',-1:'VENDA 🔴'})
        df_tr['ret_%']   = (df_tr['ret'] * 100).round(4)
        df_tr['pnl_$']   = df_tr['pnl'].round(2)
        with st.expander(f"📋 Trades ({len(bt['trades'])} total)"):
            st.dataframe(df_tr[['date','Direção','ret_%','pnl_$']].tail(50),
                         use_container_width=True)

# ── Tab 3: Indicadores ────────────────────────────────────────────────────────
with tab3:
    df_f = res['df_feat']
    col_i1, col_i2 = st.columns(2)

    with col_i1:
        # RSI
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df_f.index, y=df_f['RSI'],
                                     name="RSI", line=dict(color='#c9a84c',width=1.5)))
        fig_rsi.add_hline(y=70, line_dash="dot", line_color="#f85149",
                          annotation_text="Sobrecomprado 70")
        fig_rsi.add_hline(y=30, line_dash="dot", line_color="#3fb950",
                          annotation_text="Sobrevendido 30")
        fig_rsi.add_hline(y=50, line_dash="dash", line_color="#2a3048")
        fig_rsi.update_layout(title="RSI (14)", **base_layout(300))
        upd(fig_rsi)
        st.plotly_chart(fig_rsi, use_container_width=True)

        # Z-Score
        fig_z = go.Figure()
        fig_z.add_trace(go.Scatter(x=df_f.index, y=df_f['Z_Score'],
                                   name="Z-Score", line=dict(color='#58a6ff',width=1.5),
                                   fill='tozeroy', fillcolor='rgba(88,166,255,0.08)'))
        fig_z.add_hline(y=Z_THRESHOLD,  line_dash="dash", line_color="#f85149",
                        annotation_text=f"+{Z_THRESHOLD} (Venda)")
        fig_z.add_hline(y=-Z_THRESHOLD, line_dash="dash", line_color="#3fb950",
                        annotation_text=f"-{Z_THRESHOLD} (Compra)")
        fig_z.add_hline(y=0, line_dash="solid", line_color="#2a3048")
        fig_z.update_layout(title="Z-Score (100 períodos)", **base_layout(300))
        upd(fig_z)
        st.plotly_chart(fig_z, use_container_width=True)

    with col_i2:
        # MACD
        fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  row_heights=[0.6,0.4],
                                  subplot_titles=["Preço + MACD Signal","Histograma"])
        fig_macd.add_trace(go.Scatter(x=df_f.index, y=df_f['Ouro'],
                                      name="Ouro", line=dict(color='#c9a84c',width=1.5)), row=1,col=1)
        fig_macd.add_trace(go.Scatter(x=df_f.index, y=df_f['MACD'],
                                      name="MACD", line=dict(color='#58a6ff',width=1.2)), row=1,col=1)
        fig_macd.add_trace(go.Scatter(x=df_f.index, y=df_f['MACD_Signal'],
                                      name="Signal", line=dict(color='#d29922',width=1.2)), row=1,col=1)
        colors_macd = np.where(df_f['MACD_Hist']>=0,'#3fb950','#f85149')
        fig_macd.add_trace(go.Bar(x=df_f.index, y=df_f['MACD_Hist'],
                                   name="Hist", marker_color=colors_macd), row=2,col=1)
        fig_macd.update_layout(**base_layout(480))
        upd(fig_macd)
        st.plotly_chart(fig_macd, use_container_width=True)

        # Volatilidade
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=df_f.index, y=df_f['Volatilidade']*100,
                                     name="Volatilidade", fill='tozeroy',
                                     line=dict(color='#bc8cff',width=1.5),
                                     fillcolor='rgba(188,140,255,0.1)'))
        fig_vol.update_layout(title="Volatilidade Rolling 20 (%)", **base_layout(280))
        upd(fig_vol)
        st.plotly_chart(fig_vol, use_container_width=True)

# ── Tab 4: Checklist ──────────────────────────────────────────────────────────
with tab4:
    st.markdown("### ✅ Checklist de Condições para Abertura de Posição")
    st.markdown("*(Replica exatamente a lógica do bot_ouro_mt5 H4)*")

    conds = s['conds']
    conds_extra = {
        f"Z-Score < -{Z_THRESHOLD} (sinal Compra)": s['z'] < -Z_THRESHOLD,
        f"Z-Score > +{Z_THRESHOLD} (sinal Venda)":  s['z'] >  Z_THRESHOLD,
        "Série Estacionária (ADF)":                  s['est'],
        f"ATR > 0 ({s['atr_pct']*100:.4f}%)":       s['atr_pct'] > 0,
        f"Prob. Compra > {PROB_THRESHOLD:.0%}":      s['p_compra'] > PROB_THRESHOLD,
        f"Prob. Venda  > {PROB_THRESHOLD:.0%}":      s['p_venda']  > PROB_THRESHOLD,
    }

    for label, ok in conds_extra.items():
        emoji = "✅" if ok else "❌"
        cor   = "green" if ok else "red"
        st.markdown(f"<span style='color:{'#3fb950' if ok else '#f85149'};font-size:1.1rem;'>"
                    f"{emoji} {label}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    | Condição Geral | Status |
    |:--|:--|
    | Sinal gerado | `{s['sinal']}` |
    | Confiança | `{s['confianca']:.1%}` |
    | RSI atual | `{s['rsi']:.2f}` |
    | Z-Score | `{s['z']:+.3f}` |
    | Volatilidade | `{s['vol']:.4f}%` |
    | Estacionária | `{'Sim ✅' if s['est'] else 'Não ❌'}` |
    """)

    if s['direcao'] != 0:
        st.success(f"🟢 **SINAL ATIVO:** {s['sinal']} com {s['confianca']:.1%} de confiança.")
    else:
        st.info("🟡 **Aguardando condições ideais.** Nenhuma condição de entrada ativa no momento.")

# ── Tab 5: Modelo ML ──────────────────────────────────────────────────────────
with tab5:
    col_m1, col_m2 = st.columns(2)
    met = res['metricas']

    with col_m1:
        st.markdown("### 🧠 Random Forest — Métricas de Validação")
        st.markdown(f"""
        | Métrica | Valor |
        |:--|--:|
        | Acurácia | `{met['acc']:.1%}` |
        | Precisão (weighted) | `{met['prec']:.1%}` |
        | Recall (weighted) | `{met['rec']:.1%}` |
        | F1-Score (weighted) | `{met['f1']:.3f}` |
        | Amostras de Treino | `{met['n_train']:,}` |
        | Amostras de Val. | `{met['n_val']:,}` |
        | N° de Árvores | `100` |
        | Max Depth | `5` |
        """)

        st.markdown("### 📌 Features Utilizadas")
        df_f = res['df_feat']
        modelo = res['modelo']
        fi = pd.Series(modelo.feature_importances_, index=FEATURES).sort_values()
        fig_fi = go.Figure(go.Bar(x=fi.values, y=fi.index, orientation='h',
                                   marker_color='#c9a84c'))
        fig_fi.update_layout(title="Feature Importance (Gini)",
                              **base_layout(320))
        upd(fig_fi)
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_m2:
        st.markdown("### 📈 Distribuição das Classes (Target)")
        alvo_counts = res['df_feat']['Alvo'].value_counts().sort_index()
        labels = {1:'COMPRA',0:'NEUTRO',-1:'VENDA'}
        fig_pie = go.Figure(go.Pie(
            labels=[labels.get(i,str(i)) for i in alvo_counts.index],
            values=alvo_counts.values,
            marker_colors=['#3fb950','#8b949e','#f85149'],
            hole=0.4
        ))
        fig_pie.update_layout(paper_bgcolor=PLT_BG, font=dict(color=PLT_TEXT),
                               height=320, margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("### 📊 Distribuição do Retorno Futuro")
        ret_fut_clean = res['df_feat']['Ret_Futuro'].dropna() * 100
        fig_hist = go.Figure(go.Histogram(x=ret_fut_clean, nbinsx=60,
                                           marker_color='#c9a84c', opacity=0.8))
        fig_hist.add_vline(x=0, line_dash="dash", line_color="#f85149")
        fig_hist.update_layout(title=f"Retorno a {HORIZONTE} candles (%)",
                                **base_layout(310))
        upd(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)

# ==============================================================================
# DADOS BRUTOS
# ==============================================================================
st.markdown("---")
with st.expander("🗂️ Dados brutos (últimas 100 barras)"):
    st.dataframe(res['df_raw'].tail(100).style.format(precision=4), use_container_width=True)

with st.expander("🤖 Features calculadas (últimas 50 linhas)"):
    cols_show = ['Ouro','RSI','Z_Score','Volatilidade','ATR_Pct','Is_Stationary','Alvo']
    st.dataframe(res['df_feat'][cols_show].tail(50).style.format(precision=4),
                 use_container_width=True)

st.caption("⚠️ Robô educacional baseado em Random Forest. Resultados passados não garantem resultados futuros. Não é recomendação de investimento.")
