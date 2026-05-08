# -*- coding: utf-8 -*-
# ==============================================================================
# SISTEMA QUANTITATIVO DE ML PARA TRADING — Streamlit Dashboard
# Fusão: Stock Peer Analysis (UI) + Quant Trading ML (Algoritmo)
# ==============================================================================

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ── ML ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    from lightgbm import LGBMClassifier
    LGB_OK = True
except ImportError:
    LGB_OK = False

# ── Auto-refresh ──────────────────────────────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_OK = True
except ImportError:
    AUTOREFRESH_OK = False

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Quant Trading ML Dashboard",
    page_icon="📈",
    layout="wide",
)

# ── CSS Dark Mode ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --bg-dark:   #0b0e14;
        --bg-card:   #131720;
        --bg-input:  #1a1f2e;
        --border:    #2a3048;
        --text-main: #e8eaf6;
        --text-sub:  #8b949e;
        --primary:   #58a6ff;
        --success:   #3fb950;
        --danger:    #f85149;
        --warning:   #d29922;
        --highlight: #bc8cff;
    }
    .stApp { background-color: var(--bg-dark); color: var(--text-main); }
    section[data-testid="stSidebar"] { background-color: var(--bg-card); border-right: 1px solid var(--border); }
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    }
    .metric-card .label { color: var(--text-sub); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
    .metric-card .value { font-size: 1.9rem; font-weight: 700; }
    .metric-card .sub   { font-size: 0.75rem; color: var(--text-sub); margin-top: 4px; }
    .green { color: #3fb950; }
    .red   { color: #f85149; }
    .blue  { color: #58a6ff; }
    .purple{ color: #bc8cff; }
    h1, h2, h3 { color: var(--text-main) !important; }
    .stButton>button {
        background: linear-gradient(135deg, #58a6ff, #bc8cff);
        color: #fff; border: none; border-radius: 8px;
        font-weight: 700; padding: 0.6rem 1.8rem; font-size: 1rem;
        width: 100%; cursor: pointer;
    }
    .stButton>button:hover { opacity: 0.9; }
    div[data-testid="stMetric"] { background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 12px; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Configurações")
    st.markdown("---")

    TICKER = st.text_input("🔎 Ticker / Ativo", value="USDJPY=X",
                           help="Ex: AAPL, BTC-USD, EURUSD=X, ^GSPC, PETR4.SA")

    periodo_map = {
        "1 Ano": 1, "2 Anos": 2, "3 Anos": 3,
        "5 Anos": 5, "7 Anos": 7, "10 Anos": 10,
    }
    periodo_label = st.selectbox("📅 Período de Histórico", list(periodo_map.keys()), index=3)
    ANOS = periodo_map[periodo_label]

    TIMEFRAME = st.selectbox("⏱️ Timeframe", ["1d", "1wk"], index=0)

    st.markdown("### 🎯 Backtest")
    CAPITAL = st.number_input("Capital Inicial (USD)", value=100_000, step=10_000)
    STOP_PCT = st.slider("Stop Loss (%)", 0.5, 5.0, 2.0, 0.5) / 100
    TP_PCT   = st.slider("Take Profit (%)", 1.0, 10.0, 4.0, 0.5) / 100
    THRESH   = st.slider("Threshold de Probabilidade", 0.50, 0.95, 0.55, 0.01)

    st.markdown("### 🔄 Auto-Refresh")
    refresh_interval = st.selectbox("Intervalo de Atualização",
                                    ["Desligado", "1 min", "5 min"], index=0)

    run_btn = st.button("🚀 Executar Backtest")

# Auto-refresh
if AUTOREFRESH_OK and refresh_interval != "Desligado":
    ms = 60_000 if refresh_interval == "1 min" else 300_000
    st_autorefresh(interval=ms, key="autorefresh")

# ==============================================================================
# TÍTULO
# ==============================================================================
st.markdown("# 📊 Quant Trading ML Dashboard")
st.markdown(f"**Ativo:** `{TICKER.upper()}` &nbsp;|&nbsp; **Período:** {periodo_label} &nbsp;|&nbsp; **Timeframe:** {TIMEFRAME}")
st.markdown("---")

# ==============================================================================
# FUNÇÕES CORE
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
    if 'Volume' not in df.columns:
        df['Volume'] = 0
    df['Volume'] = df['Volume'].fillna(0)
    df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
    df = df.sort_index()
    return df


def engenharia_de_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df['Close']
    h = df['High']
    l = df['Low']
    o = df['Open']
    v = df['Volume'].replace(0, np.nan)

    # Médias móveis
    for p in [5, 10, 20, 50, 100, 200]:
        df[f'sma_{p}']  = c.rolling(p).mean()
        df[f'ema_{p}']  = c.ewm(span=p, adjust=False).mean()
    for p in [5, 10, 20, 50, 200]:
        df[f'ratio_sma_{p}'] = c / df[f'sma_{p}'] - 1

    df['cruz_5_20']   = (df['sma_5']  > df['sma_20']).astype(int)
    df['cruz_20_50']  = (df['sma_20'] > df['sma_50']).astype(int)
    df['cruz_50_200'] = (df['sma_50'] > df['sma_200']).astype(int)

    # RSI
    for p in [7, 14, 21]:
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(p).mean()
        loss  = (-delta.clip(upper=0)).rolling(p).mean()
        rs    = gain / loss.replace(0, np.nan)
        df[f'rsi_{p}'] = 100 - (100 / (1 + rs))

    # ROC
    for p in [5, 10, 20, 60]:
        df[f'roc_{p}'] = c.pct_change(p) * 100

    # Estocástico
    for p in [14, 21]:
        lmin = l.rolling(p).min()
        hmax = h.rolling(p).max()
        df[f'stoch_k_{p}'] = 100 * (c - lmin) / (hmax - lmin).replace(0, np.nan)
        df[f'stoch_d_{p}'] = df[f'stoch_k_{p}'].rolling(3).mean()

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df['macd']        = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']

    # ATR
    tr = pd.concat([h - l,
                    (h - c.shift()).abs(),
                    (l - c.shift()).abs()], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_norm'] = df['atr_14'] / c

    # Bollinger
    for p in [20]:
        mid  = c.rolling(p).mean()
        std  = c.rolling(p).std()
        df[f'bb_upper_{p}'] = mid + 2 * std
        df[f'bb_lower_{p}'] = mid - 2 * std
        df[f'bb_width_{p}'] = (df[f'bb_upper_{p}'] - df[f'bb_lower_{p}']) / mid
        df[f'bb_pos_{p}']   = (c - df[f'bb_lower_{p}']) / (df[f'bb_upper_{p}'] - df[f'bb_lower_{p}']).replace(0, np.nan)

    # Z-score
    for p in [20, 60]:
        df[f'zscore_{p}'] = (c - c.rolling(p).mean()) / c.rolling(p).std()

    # Retornos defasados
    for lag in [1, 2, 3, 5, 10]:
        df[f'ret_lag_{lag}'] = c.pct_change(lag)

    # Volatilidade rolling
    for p in [5, 10, 20]:
        df[f'vol_roll_{p}'] = c.pct_change().rolling(p).std() * np.sqrt(252)

    # Candlestick
    df['corpo']        = (c - o) / o
    df['sombra_sup']   = (h - c.clip(lower=o)) / o
    df['sombra_inf']   = (c.clip(upper=o) - l) / o
    df['gap']          = (o - c.shift(1)) / c.shift(1)
    df['alta_seguida'] = (c > c.shift(1)).astype(int)

    # Volume relativo
    df['vol_rel'] = v / v.rolling(20).mean()

    # Skew & Kurt rolling
    df['skew_20']  = c.pct_change().rolling(20).skew()
    df['kurt_20']  = c.pct_change().rolling(20).kurt()

    # Target
    df['ret_futuro'] = np.log(c.shift(-1) / c)
    df['target']     = (df['ret_futuro'] > 0).astype(int)

    return df


def processar_estrategia(df: pd.DataFrame, thresh: float = 0.55):
    """
    Função principal: feature engineering → treino → backtest.
    Retorna métricas, df com sinais e equity curve.
    """
    df = engenharia_de_features(df)
    df = df.iloc[:-1]  # Remove última linha (target inválido)

    # Remove apenas colunas com >60% de NaN antes de dropar linhas
    thresh_nan = 0.6
    df = df.loc[:, df.isnull().mean() < thresh_nan]

    # Drop linhas com NaN restantes
    df = df.dropna()

    MIN_AMOSTRAS = 120  # mínimo para splits terem sentido
    if len(df) < MIN_AMOSTRAS:
        raise ValueError(
            f"Dados insuficientes após feature engineering: apenas {len(df)} linhas. "
            f"Aumente o período histórico (mínimo recomendado: 3 anos) ou troque o ativo."
        )

    feature_cols = [c for c in df.columns if c not in
                    ['Open','High','Low','Close','Volume',
                     'target','ret_futuro']]

    X = df[feature_cols].values
    y = df['target'].values
    dates = df.index

    # Split temporal 70/15/15
    n = len(df)
    t1, t2 = int(n * 0.70), int(n * 0.85)

    # Garante tamanho mínimo em cada split
    MIN_SPLIT = 20
    if t1 < MIN_SPLIT or (t2 - t1) < MIN_SPLIT or (n - t2) < MIN_SPLIT:
        raise ValueError(
            f"Splits muito pequenos (treino={t1}, val={t2-t1}, teste={n-t2}). "
            f"Aumente o período histórico."
        )

    X_tr, y_tr = X[:t1], y[:t1]
    X_val, y_val = X[t1:t2], y[t1:t2]
    X_te, y_te = X[t2:], y[t2:]

    # Scaler
    scaler = RobustScaler()
    X_tr  = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te  = scaler.transform(X_te)

    # Modelos
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=6,
                                      random_state=42, n_jobs=-1)),
        ('lr', LogisticRegression(max_iter=1000, C=0.5, random_state=42)),
    ]
    if XGB_OK:
        estimators.append(('xgb', XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            use_label_encoder=False, eval_metric='logloss',
            random_state=42, verbosity=0)))
    if LGB_OK:
        estimators.append(('lgb', LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=42, verbosity=-1)))

    model = VotingClassifier(estimators=estimators, voting='soft')
    model.fit(X_tr, y_tr)

    # Métricas (conjunto de teste)
    proba_te = model.predict_proba(X_te)[:, 1]
    pred_te  = (proba_te >= thresh).astype(int)

    acc  = accuracy_score(y_te, pred_te)
    prec = precision_score(y_te, pred_te, zero_division=0)
    rec  = recall_score(y_te, pred_te, zero_division=0)
    f1   = f1_score(y_te, pred_te, zero_division=0)

    # Backtest no conjunto de teste
    close_te   = df['Close'].values[t2:]
    ret_fut_te = df['ret_futuro'].values[t2:]
    dates_te   = dates[t2:]

    capital   = 1.0
    equity    = [capital]
    trades    = []
    SPREAD    = 0.0002
    SLIP      = 0.0001
    COST      = 0.0001

    for i, (p, ret, dt) in enumerate(zip(proba_te, ret_fut_te, dates_te)):
        if p >= thresh:       # Sinal de compra
            trade_ret = ret - SPREAD - SLIP - COST
            # Aplica stop/tp simplificado
            if trade_ret < -STOP_PCT:
                trade_ret = -STOP_PCT
            elif trade_ret > TP_PCT:
                trade_ret = TP_PCT
            capital *= (1 + trade_ret)
            trades.append({'date': dt, 'ret': trade_ret, 'signal': 1, 'prob': p})
        equity.append(capital)

    equity = np.array(equity)
    ret_acum   = (equity[-1] - 1) * 100
    max_dd     = _max_drawdown(equity)
    sharpe     = _sharpe(equity)
    win_rate   = np.mean([t['ret'] > 0 for t in trades]) * 100 if trades else 0.0
    n_trades   = len(trades)

    # DataFrame de equity para plot
    eq_df = pd.DataFrame({
        'date':   list(dates_te) + [dates_te[-1]],  # mesmo tamanho
        'equity': equity[:len(dates_te)+1]
    })
    # Ajuste de tamanho
    min_len = min(len(dates_te), len(equity))
    eq_df = pd.DataFrame({'date': dates_te[:min_len], 'equity': equity[:min_len]})

    df_out = df.iloc[t2:].copy()
    df_out['proba'] = proba_te
    df_out['sinal'] = pred_te

    return {
        'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
        'ret_acum': ret_acum, 'max_dd': max_dd,
        'sharpe': sharpe, 'win_rate': win_rate, 'n_trades': n_trades,
        'equity': equity, 'eq_df': eq_df,
        'df_out': df_out, 'features': feature_cols,
        'n_train': t1, 'n_val': t2 - t1, 'n_test': n - t2,
        'close_te': close_te,
    }


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd   = (equity - peak) / peak
    return float(dd.min() * 100)


def _sharpe(equity: np.ndarray, rf: float = 0.0) -> float:
    rets = np.diff(equity) / equity[:-1]
    if rets.std() == 0:
        return 0.0
    return float((rets.mean() - rf / 252) / rets.std() * np.sqrt(252))


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================

if run_btn or 'results' not in st.session_state:
    with st.spinner(f"⏳ Baixando dados de `{TICKER}` e treinando modelos..."):
        try:
            df_raw = baixar_dados(TICKER.upper(), ANOS, TIMEFRAME)
            if len(df_raw) < 200:
                st.error("⚠️ Dados insuficientes. Tente um período maior ou outro ativo.")
                st.stop()
            results = processar_estrategia(df_raw, thresh=THRESH)
            results['df_raw'] = df_raw
            st.session_state['results'] = results
            st.session_state['ticker']  = TICKER.upper()
        except Exception as e:
            st.error(f"Erro: {e}")
            st.stop()

results = st.session_state.get('results')
if not results:
    st.info("Configure os parâmetros na barra lateral e clique em **🚀 Executar Backtest**.")
    st.stop()

# ==============================================================================
# MÉTRICAS — CARDS DE DESTAQUE
# ==============================================================================
st.markdown("## 📈 Resultados do Backtest")

r = results
c1, c2, c3, c4 = st.columns(4)

def card(col, label, value, css_class, sub=""):
    col.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value {css_class}">{value}</div>
        <div class="sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

ret_class  = "green" if r['ret_acum'] >= 0 else "red"
dd_class   = "red"
shr_class  = "green" if r['sharpe'] >= 1 else ("warning" if r['sharpe'] >= 0 else "red")
wr_class   = "green" if r['win_rate'] >= 50 else "red"

card(c1, "Retorno Acumulado", f"{r['ret_acum']:+.2f}%", ret_class, f"{r['n_trades']} trades")
card(c2, "Sharpe Ratio",      f"{r['sharpe']:.2f}",      shr_class,  "Anualizado")
card(c3, "Win Rate",          f"{r['win_rate']:.1f}%",   wr_class,   f"Acurácia: {r['acc']:.1%}")
card(c4, "Max Drawdown",      f"{r['max_dd']:.2f}%",     dd_class,   "Pior queda")

st.markdown("<br>", unsafe_allow_html=True)

# Sub-métricas
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Precisão",  f"{r['prec']:.1%}")
m2.metric("Recall",    f"{r['rec']:.1%}")
m3.metric("F1-Score",  f"{r['f1']:.3f}")
m4.metric("N° Trades", r['n_trades'])
m5.metric("Nº Features", len(r['features']))

st.markdown("---")

# ==============================================================================
# GRÁFICOS
# ==============================================================================
tab1, tab2, tab3 = st.tabs(["📉 Equity Curve & Preço", "🔬 Sinais ML", "📊 Distribuição"])

with tab1:
    df_raw = results['df_raw']
    df_out = results['df_out']
    eq     = results['equity']

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45],
                        subplot_titles=["Preço de Fechamento", "Equity Curve (Backtest — Conjunto de Teste)"])

    # Preço
    fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw['Close'],
                             name="Preço", line=dict(color='#58a6ff', width=1.5)), row=1, col=1)
    # Sinais compra
    buys = df_out[df_out['sinal'] == 1]
    fig.add_trace(go.Scatter(x=buys.index, y=buys['Close'],
                             mode='markers', name="Sinal Compra",
                             marker=dict(color='#3fb950', size=5, symbol='triangle-up')), row=1, col=1)

    # Equity
    eq_df = results['eq_df']
    fig.add_trace(go.Scatter(x=eq_df['date'], y=eq_df['equity'] * CAPITAL,
                             name="Equity (USD)", fill='tozeroy',
                             line=dict(color='#bc8cff', width=2),
                             fillcolor='rgba(188,140,255,0.1)'), row=2, col=1)
    fig.add_hline(y=CAPITAL, line_dash="dash", line_color="#8b949e", row=2, col=1)

    fig.update_layout(
        height=600, paper_bgcolor='#0b0e14', plot_bgcolor='#131720',
        font=dict(color='#e8eaf6'), legend=dict(bgcolor='#131720'),
        margin=dict(l=40, r=20, t=40, b=20),
    )
    fig.update_xaxes(gridcolor='#2a3048', zeroline=False)
    fig.update_yaxes(gridcolor='#2a3048', zeroline=False)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    df_s = results['df_out'].copy()

    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.6, 0.4],
                         subplot_titles=["Probabilidade Predita (ML)", "RSI 14"])

    fig2.add_trace(go.Scatter(x=df_s.index, y=df_s['proba'],
                              name="Prob. Alta", line=dict(color='#58a6ff', width=1.5)), row=1, col=1)
    fig2.add_hline(y=THRESH, line_dash="dash", line_color="#d29922",
                   annotation_text=f"Threshold {THRESH:.0%}", row=1, col=1)

    if 'rsi_14' in df_s.columns:
        fig2.add_trace(go.Scatter(x=df_s.index, y=df_s['rsi_14'],
                                  name="RSI 14", line=dict(color='#bc8cff', width=1.5)), row=2, col=1)
        fig2.add_hline(y=70, line_dash="dot", line_color="#f85149", row=2, col=1)
        fig2.add_hline(y=30, line_dash="dot", line_color="#3fb950", row=2, col=1)

    fig2.update_layout(
        height=500, paper_bgcolor='#0b0e14', plot_bgcolor='#131720',
        font=dict(color='#e8eaf6'), legend=dict(bgcolor='#131720'),
        margin=dict(l=40, r=20, t=40, b=20),
    )
    fig2.update_xaxes(gridcolor='#2a3048')
    fig2.update_yaxes(gridcolor='#2a3048')
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    col_a, col_b = st.columns(2)

    with col_a:
        # Distribuição de probabilidades
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=results['df_out']['proba'],
                                    nbinsx=40, name="Prob. Predita",
                                    marker_color='#58a6ff', opacity=0.8))
        fig3.add_vline(x=THRESH, line_dash="dash", line_color="#d29922")
        fig3.update_layout(
            title="Distribuição de Probabilidades",
            height=350, paper_bgcolor='#0b0e14', plot_bgcolor='#131720',
            font=dict(color='#e8eaf6'), showlegend=False,
            margin=dict(l=40, r=20, t=50, b=20),
        )
        fig3.update_xaxes(gridcolor='#2a3048')
        fig3.update_yaxes(gridcolor='#2a3048')
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        # Drawdown ao longo do tempo
        eq = results['equity']
        peak = np.maximum.accumulate(eq)
        dd   = (eq - peak) / peak * 100
        eq_df = results['eq_df']
        dd_trim = dd[:len(eq_df)]

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=eq_df['date'][:len(dd_trim)], y=dd_trim,
                                  fill='tozeroy', name="Drawdown (%)",
                                  line=dict(color='#f85149', width=1.5),
                                  fillcolor='rgba(248,81,73,0.15)'))
        fig4.update_layout(
            title="Drawdown ao Longo do Tempo",
            height=350, paper_bgcolor='#0b0e14', plot_bgcolor='#131720',
            font=dict(color='#e8eaf6'), showlegend=False,
            margin=dict(l=40, r=20, t=50, b=20),
        )
        fig4.update_xaxes(gridcolor='#2a3048')
        fig4.update_yaxes(gridcolor='#2a3048')
        st.plotly_chart(fig4, use_container_width=True)

# ==============================================================================
# DADOS BRUTOS (EXPANSÍVEL)
# ==============================================================================
st.markdown("---")
with st.expander("🗂️ Ver dados brutos do ativo"):
    st.dataframe(results['df_raw'].tail(100).style.format(precision=4),
                 use_container_width=True)

with st.expander("🤖 Ver sinais do modelo (conjunto de teste)"):
    st.dataframe(results['df_out'][['Close','proba','sinal','rsi_14','macd']].tail(60)
                 .style.format(precision=4), use_container_width=True)

st.caption("⚠️ Este dashboard é educacional. Não constitui recomendação de investimento.")
