# -*- coding: utf-8 -*-
# ==============================================================================
# SISTEMA QUANTITATIVO DE ML PARA TRADING — Streamlit Dashboard
# Fusão completa: Stock Peer Analysis (UI) + Quant ML System v3.0 (Algoritmo)
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
from scipy.stats import norm as sci_norm

# ── ML ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve,
                              confusion_matrix)
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

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

st.markdown("""
<style>
    :root {
        --bg-dark:#0b0e14; --bg-card:#131720; --border:#2a3048;
        --text-main:#e8eaf6; --text-sub:#8b949e;
        --primary:#58a6ff; --success:#3fb950; --danger:#f85149;
        --warning:#d29922; --highlight:#bc8cff;
    }
    .stApp { background-color: var(--bg-dark); color: var(--text-main); }
    section[data-testid="stSidebar"] { background-color: var(--bg-card); border-right:1px solid var(--border); }
    .metric-card { background:var(--bg-card); border:1px solid var(--border); border-radius:12px;
        padding:20px 24px; text-align:center; box-shadow:0 4px 16px rgba(0,0,0,0.4); }
    .metric-card .label { color:var(--text-sub); font-size:0.78rem; text-transform:uppercase;
        letter-spacing:1px; margin-bottom:6px; }
    .metric-card .value { font-size:1.9rem; font-weight:700; }
    .metric-card .sub   { font-size:0.73rem; color:var(--text-sub); margin-top:4px; }
    .green{color:#3fb950;} .red{color:#f85149;} .blue{color:#58a6ff;} .purple{color:#bc8cff;}
    h1,h2,h3{color:var(--text-main)!important;}
    .stButton>button { background:linear-gradient(135deg,#58a6ff,#bc8cff); color:#fff;
        border:none; border-radius:8px; font-weight:700; padding:0.6rem 1.8rem;
        font-size:1rem; width:100%; cursor:pointer; }
    .stButton>button:hover{opacity:0.9;}
    div[data-testid="stMetric"]{background:var(--bg-card);border:1px solid var(--border);
        border-radius:10px;padding:12px;}
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
    periodo_map = {"3 Anos":3,"5 Anos":5,"7 Anos":7,"10 Anos":10}
    periodo_label = st.selectbox("📅 Período de Histórico", list(periodo_map.keys()), index=1)
    ANOS = periodo_map[periodo_label]
    TIMEFRAME = st.selectbox("⏱️ Timeframe", ["1d","1wk"], index=0)

    st.markdown("### 🎯 Backtest")
    CAPITAL    = st.number_input("Capital Inicial (USD)", value=100_000, step=10_000)
    STOP_PCT   = st.slider("Stop Loss (%)", 0.5, 5.0, 2.0, 0.5) / 100
    TP_PCT     = st.slider("Take Profit (%)", 1.0, 10.0, 4.0, 0.5) / 100
    THRESH     = st.slider("Threshold de Probabilidade", 0.50, 0.95, 0.55, 0.01)
    SPREAD     = 0.0002
    SLIPPAGE   = 0.0001
    RISCO_TRADE= 0.02

    st.markdown("### 🔄 Auto-Refresh")
    refresh_interval = st.selectbox("Intervalo", ["Desligado","1 min","5 min"], index=0)
    run_btn = st.button("🚀 Executar Backtest")

if AUTOREFRESH_OK and refresh_interval != "Desligado":
    ms = 60_000 if refresh_interval == "1 min" else 300_000
    st_autorefresh(interval=ms, key="autorefresh")

st.markdown("# 📊 Quant Trading ML Dashboard")
st.markdown(f"**Ativo:** `{TICKER.upper()}` &nbsp;|&nbsp; **Período:** {periodo_label} &nbsp;|&nbsp; **Timeframe:** {TIMEFRAME}")
st.markdown("---")

# ==============================================================================
# PASSO 2 — DOWNLOAD DE DADOS
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
    df.columns = [c if c != 'Nan' else '_drop' for c in df.columns]
    df.drop(columns=[c for c in df.columns if c == '_drop'], errors='ignore', inplace=True)
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

# ==============================================================================
# PASSO 5 — FEATURE ENGINEERING COMPLETO (todos os 10 grupos do Colab)
# ==============================================================================
def engenharia_de_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df['Close']; h = df['High']; l = df['Low']
    o = df['Open'];  v = df['Volume'].replace(0, np.nan)
    ret = c.pct_change()

    # 1. Médias Móveis
    for p in [5,10,20,50,100,200]:
        df[f'sma_{p}'] = c.rolling(p).mean()
        df[f'ema_{p}'] = c.ewm(span=p, adjust=False).mean()
    for p in [5,10,20,50,200]:
        df[f'ratio_sma_{p}'] = c / df[f'sma_{p}'] - 1
    df['cruz_5_20']   = (df['sma_5']  > df['sma_20']).astype(int)
    df['cruz_20_50']  = (df['sma_20'] > df['sma_50']).astype(int)
    df['cruz_50_200'] = (df['sma_50'] > df['sma_200']).astype(int)
    typical = (h + l + c) / 3
    df['vwap'] = (typical * v).rolling(20).sum() / v.rolling(20).sum()
    df['ratio_vwap'] = c / df['vwap'] - 1

    # 2. Momentum
    for p in [7,14,21]:
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(p).mean()
        loss  = (-delta.clip(upper=0)).rolling(p).mean()
        rs    = gain / loss.replace(0, np.nan)
        df[f'rsi_{p}']      = 100 - (100 / (1 + rs))
        df[f'rsi_{p}_norm'] = df[f'rsi_{p}'] / 50 - 1
    for p in [5,10,20,60]:
        df[f'roc_{p}'] = c.pct_change(p) * 100
    for p in [5,10,20]:
        df[f'mom_{p}']      = c - c.shift(p)
        df[f'mom_{p}_norm'] = df[f'mom_{p}'] / c.shift(p)
    for p in [14,21]:
        lmin = l.rolling(p).min()
        hmax = h.rolling(p).max()
        denom = (hmax - lmin).replace(0, np.nan)
        df[f'stoch_k_{p}']    = 100 * (c - lmin) / denom
        df[f'stoch_d_{p}']    = df[f'stoch_k_{p}'].rolling(3).mean()
        df[f'stoch_diff_{p}'] = df[f'stoch_k_{p}'] - df[f'stoch_d_{p}']

    # 3. Volatilidade
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    for p in [7,14,21]:
        df[f'atr_{p}']      = true_range.ewm(span=p, adjust=False).mean()
        df[f'atr_{p}_norm'] = df[f'atr_{p}'] / c
    for p in [5,10,20,60]:
        df[f'vol_{p}'] = ret.rolling(p).std() * np.sqrt(252)
    for p in [20]:
        mid = c.rolling(p).mean()
        std = c.rolling(p).std()
        df[f'bb_upper_{p}'] = mid + 2 * std
        df[f'bb_lower_{p}'] = mid - 2 * std
        df[f'bb_width_{p}'] = (df[f'bb_upper_{p}'] - df[f'bb_lower_{p}']) / mid
        df[f'bb_pos_{p}']   = (c - df[f'bb_lower_{p}']) / (
            (df[f'bb_upper_{p}'] - df[f'bb_lower_{p}']).replace(0, np.nan))
    df['vol_regime'] = df['vol_5'] / df['vol_20']

    # 4. Tendência
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df['macd']        = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']
    df['macd_norm']   = df['macd'] / c
    for p in [14]:
        dm_plus  = h.diff().clip(lower=0)
        dm_minus = (-l.diff()).clip(lower=0)
        tr_s     = true_range.ewm(span=p, adjust=False).mean()
        di_p     = 100 * dm_plus.ewm(span=p, adjust=False).mean() / tr_s.replace(0, np.nan)
        di_m     = 100 * dm_minus.ewm(span=p, adjust=False).mean() / tr_s.replace(0, np.nan)
        dx       = 100 * (di_p - di_m).abs() / (di_p + di_m + 1e-9)
        df[f'adx_{p}']      = dx.ewm(span=p, adjust=False).mean()
        df[f'di_plus_{p}']  = di_p
        df[f'di_minus_{p}'] = di_m
        df[f'di_diff_{p}']  = di_p - di_m
    for p in [20,50]:
        df[f'ema_{p}_slope'] = df[f'ema_{p}'].diff(5) / df[f'ema_{p}'].shift(5)

    # 5. Mean Reversion
    for p in [20,60]:
        mu = c.rolling(p).mean()
        sg = c.rolling(p).std()
        df[f'zscore_{p}'] = (c - mu) / sg.replace(0, np.nan)
    for p in [20,50,200]:
        df[f'dist_ema_{p}'] = (c / df[f'ema_{p}'] - 1) * 100

    # 6. Volume
    df['vol_rel_20'] = v / v.rolling(20).mean()
    df['vol_rel_50'] = v / v.rolling(50).mean()
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    df['obv']       = obv
    df['obv_sma20'] = obv.rolling(20).mean()
    df['obv_ratio'] = obv / (df['obv_sma20'] + 1e-9)
    mf_vol = ((c - l) - (h - c)) / (h - l + 1e-9) * v
    df['mf_20'] = mf_vol.rolling(20).sum() / v.rolling(20).sum()
    df['price_vol']      = (c * v).rolling(10).mean()
    df['price_vol_norm'] = df['price_vol'] / df['price_vol'].rolling(60).mean()

    # 7. Candlestick
    df['corpo']         = (c - o).abs()
    df['corpo_pct']     = df['corpo'] / o.clip(lower=1e-9)
    df['sombra_sup']    = h - pd.concat([c, o], axis=1).max(axis=1)
    df['sombra_inf']    = pd.concat([c, o], axis=1).min(axis=1) - l
    df['sombra_sup_pct']= df['sombra_sup'] / o.clip(lower=1e-9)
    df['sombra_inf_pct']= df['sombra_inf'] / o.clip(lower=1e-9)
    df['range']         = h - l
    df['range_pct']     = df['range'] / o.clip(lower=1e-9)
    df['gap']           = (o - c.shift(1)) / c.shift(1).clip(lower=1e-9)
    df['direcao_candle']= np.sign(c - o)
    for n in [3,5]:
        dir_s = (c > o).astype(int)
        df[f'seq_alta_{n}'] = dir_s.rolling(n).sum()
        df[f'pct_alta_{n}'] = df[f'seq_alta_{n}'] / n

    # 8. Lags
    for lag in [1,2,3,5,10,20]:
        df[f'ret_lag_{lag}'] = ret.shift(lag)
    for lag in [1,5,10]:
        df[f'vol20_lag_{lag}'] = df['vol_20'].shift(lag)
    for lag in [1,3,5]:
        df[f'rsi14_lag_{lag}'] = df['rsi_14'].shift(lag)
    for lag in [1,3]:
        df[f'macd_lag_{lag}'] = df['macd'].shift(lag)

    # 9. Estatísticas Rolling
    for p in [20,60]:
        df[f'skew_{p}']  = ret.rolling(p).skew()
        df[f'kurt_{p}']  = ret.rolling(p).kurt()
        df[f'maxdd_{p}'] = c.rolling(p).apply(
            lambda x: (x[-1]/x.max()-1) if len(x)>0 else 0, raw=True)
        df[f'autocorr_{p}'] = ret.rolling(p).apply(
            lambda x: pd.Series(x).autocorr(lag=1) if len(x)>1 else 0, raw=True)

    # 10. Regimes de Mercado
    df['regime_tendencia'] = (df['adx_14'] > 25).astype(int)
    df['regime_alta_vol']  = (df['vol_regime'] > 1.2).astype(int)
    df['regime_bull'] = ((c > df['sma_50']) & (c > df['sma_200'])).astype(int)
    df['regime_bear'] = ((c < df['sma_50']) & (c < df['sma_200'])).astype(int)

    def hurst_rs(ts):
        if len(ts) < 10 or ts.std() == 0: return 0.5
        mean = ts.mean(); dev = ts - mean; cumdev = np.cumsum(dev)
        R = cumdev.max() - cumdev.min(); S = ts.std()
        return np.log(R/S) / np.log(len(ts)) if S > 0 else 0.5

    df['hurst_60'] = ret.rolling(60).apply(hurst_rs, raw=True)

    # Features de tempo
    df['dia_semana'] = df.index.dayofweek
    df['dia_mes']    = df.index.day
    df['mes']        = df.index.month
    df['trimestre']  = df.index.quarter
    df['fim_semana'] = (df.index.dayofweek >= 3).astype(int)
    df['inicio_mes'] = (df.index.day <= 5).astype(int)
    df['fim_mes']    = (df.index.day >= 25).astype(int)

    # Target
    df['ret_futuro'] = np.log(c.shift(-1) / c)
    df['target']     = (df['ret_futuro'] > 0).astype(int)
    df['ret_atual']  = np.log(c / c.shift(1))

    return df

# ==============================================================================
# PASSO 6 — SELEÇÃO DE FEATURES (Mutual Information)
# ==============================================================================
COLS_NAO_FEATURE = ['Open','High','Low','Close','Volume','target',
                    'ret_futuro','ret_atual']

def selecionar_features(df: pd.DataFrame, max_features: int = 60,
                         corr_max: float = 0.95) -> list:
    candidatas = [c for c in df.columns if c not in COLS_NAO_FEATURE]
    df_f = df[candidatas + ['target']].copy()

    # 1. Remover colunas com >30% NaN
    nan_pct = df_f.isnull().mean()
    ok = [f for f in nan_pct[nan_pct < 0.30].index if f != 'target']

    # 2. Dropar linhas NaN e checar variância
    df_c = df_f[ok + ['target']].dropna()
    if len(df_c) < 50:
        return ok[:max_features]

    var = df_c[ok].var()
    ok = var[var > 1e-10].index.tolist()

    # 3. Remover multicolinearidade
    corr = df_c[ok].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    remover = {col for col in upper.columns if any(upper[col] > corr_max)}
    ok = [f for f in ok if f not in remover]

    if not ok:
        return []

    # 4. Mutual Information ranking
    mi = mutual_info_classif(df_c[ok].values, df_c['target'].values, random_state=42)
    mi_rank = pd.Series(mi, index=ok).sort_values(ascending=False)
    return mi_rank.head(max_features).index.tolist(), mi_rank

# ==============================================================================
# FUNÇÃO PRINCIPAL: processar_estrategia(df)
# ==============================================================================
def processar_estrategia(df_raw: pd.DataFrame,
                          thresh: float = 0.55,
                          capital: float = 100_000,
                          stop_pct: float = 0.02,
                          tp_pct: float = 0.04,
                          spread: float = 0.0002,
                          slippage: float = 0.0001,
                          risco_trade: float = 0.02) -> dict:

    # ── Engenharia de features ────────────────────────────────────────────────
    df = engenharia_de_features(df_raw)
    df = df.iloc[:-1]   # remove última linha (target inválido)

    # ── Limpeza defensiva ─────────────────────────────────────────────────────
    df = df.loc[:, df.isnull().mean() < 0.60]   # remove colunas muito NaN
    df = df.dropna()

    if len(df) < 150:
        raise ValueError(
            f"Dados insuficientes após feature engineering: {len(df)} linhas. "
            "Aumente o período histórico (mín. 5 anos em 1d ou 3 anos em 1wk)."
        )

    # ── Seleção de features ───────────────────────────────────────────────────
    result_feat = selecionar_features(df, max_features=60)
    if isinstance(result_feat, tuple):
        feature_cols, mi_rank = result_feat
    else:
        feature_cols, mi_rank = result_feat, pd.Series(dtype=float)

    if not feature_cols:
        feature_cols = [c for c in df.columns if c not in COLS_NAO_FEATURE][:40]

    X = df[feature_cols].values
    y = df['target'].values
    dates = df.index
    closes = df['Close'].values
    ret_fut = df['ret_futuro'].values

    # ── Split temporal 70/15/15 ───────────────────────────────────────────────
    n = len(df)
    t1, t2 = int(n * 0.70), int(n * 0.85)

    MIN = 30
    if t1 < MIN or (t2-t1) < MIN or (n-t2) < MIN:
        raise ValueError(
            f"Splits muito pequenos (treino={t1}, val={t2-t1}, teste={n-t2}). "
            "Aumente o período histórico."
        )

    X_tr, y_tr   = X[:t1],    y[:t1]
    X_val, y_val = X[t1:t2],  y[t1:t2]
    X_te, y_te   = X[t2:],    y[t2:]
    ret_te        = ret_fut[t2:]
    cl_te         = closes[t2:]
    dates_te      = dates[t2:]

    # ── Normalização (fit apenas no treino) ───────────────────────────────────
    scaler = RobustScaler()
    X_tr  = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te  = scaler.transform(X_te)

    # ── Passo 10 — Treinamento de múltiplos modelos ───────────────────────────
    estimadores = [
        ('rf',  RandomForestClassifier(n_estimators=200, max_depth=6,
                                       min_samples_leaf=10, random_state=42, n_jobs=-1)),
        ('lr',  LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
        ('dt',  DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42)),
    ]
    if XGB_OK:
        estimadores.append(('xgb', XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0)))
    if LGB_OK:
        estimadores.append(('lgb', LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            random_state=42, n_jobs=-1, verbosity=-1)))

    # Treinar cada modelo individualmente para comparação
    modelos_resultado = []
    modelos_treinados = {}
    for nome, mod in estimadores:
        mod.fit(X_tr, y_tr)
        yp  = mod.predict(X_te)
        ypr = mod.predict_proba(X_te)[:,1]
        modelos_resultado.append({
            'Modelo': nome.upper(),
            'Acc':    accuracy_score(y_te, yp),
            'F1':     f1_score(y_te, yp, zero_division=0),
            'ROC':    roc_auc_score(y_te, ypr),
            'Prec':   precision_score(y_te, yp, zero_division=0),
            'Rec':    recall_score(y_te, yp, zero_division=0),
        })
        modelos_treinados[nome] = mod

    df_modelos = pd.DataFrame(modelos_resultado).sort_values('ROC', ascending=False)

    # ── Passo 12 — Ensemble VotingClassifier (soft) ───────────────────────────
    model = VotingClassifier(estimators=estimadores, voting='soft')
    model.fit(X_tr, y_tr)

    proba_te = model.predict_proba(X_te)[:,1]
    pred_te  = (proba_te >= thresh).astype(int)

    acc  = accuracy_score(y_te, pred_te)
    prec = precision_score(y_te, pred_te, zero_division=0)
    rec  = recall_score(y_te, pred_te, zero_division=0)
    f1   = f1_score(y_te, pred_te, zero_division=0)
    roc  = roc_auc_score(y_te, proba_te)
    fpr, tpr, _ = roc_curve(y_te, proba_te)
    cm   = confusion_matrix(y_te, pred_te)

    # ── Passo 14 — Backtest Realista ──────────────────────────────────────────
    cap_bt    = capital
    equity    = [cap_bt]
    trades    = []

    for i, (p, ret, dt) in enumerate(zip(proba_te, ret_te, dates_te)):
        sinal = 0
        if p >= thresh:          sinal =  1
        elif p <= (1 - thresh):  sinal = -1

        if sinal == 0:
            equity.append(cap_bt)
            continue

        custo  = (spread + slippage) * cap_bt
        tam    = cap_bt * risco_trade
        pnl_b  = tam * sinal * ret

        # Stop / Take-Profit
        if sinal == 1:
            if ret < -stop_pct:  pnl_b = -tam * stop_pct
            elif ret > tp_pct:   pnl_b =  tam * tp_pct
        else:
            if ret >  stop_pct:  pnl_b = -tam * stop_pct
            elif ret < -tp_pct:  pnl_b =  tam * tp_pct

        pnl = pnl_b - custo
        cap_bt += pnl
        equity.append(cap_bt)
        trades.append({'date': dt, 'ret': pnl/capital, 'sinal': sinal, 'prob': p, 'pnl': pnl})

    equity = np.array(equity)

    # ── Métricas financeiras ──────────────────────────────────────────────────
    ret_acum = (equity[-1] / capital - 1) * 100
    peak     = np.maximum.accumulate(equity)
    dd_arr   = (equity - peak) / peak
    max_dd   = float(dd_arr.min() * 100)
    rets_eq  = np.diff(equity) / equity[:-1]
    sharpe   = float((rets_eq.mean() / rets_eq.std() * np.sqrt(252))
                     if rets_eq.std() > 0 else 0)
    dn       = rets_eq[rets_eq < 0]
    sortino  = float((rets_eq.mean() / dn.std() * np.sqrt(252))
                     if len(dn) > 0 and dn.std() > 0 else 0)
    win_rate = float(np.mean([t['ret'] > 0 for t in trades]) * 100) if trades else 0.0
    n_trades = len(trades)

    # ── Passo 16 — Análise de Threshold ──────────────────────────────────────
    thresholds = np.arange(0.50, 0.76, 0.02)
    thr_data   = []
    for thr in thresholds:
        yp_t = (proba_te >= thr).astype(int)
        ns   = int(yp_t.sum())
        if ns < 5: continue
        thr_data.append({
            'thr':      thr,
            'f1':       f1_score(y_te, yp_t, zero_division=0),
            'acc':      accuracy_score(y_te, yp_t),
            'prec':     precision_score(y_te, yp_t, zero_division=0),
            'n_sinais': ns,
        })
    df_thr = pd.DataFrame(thr_data)

    # ── Passo 17 — Overfitting check ─────────────────────────────────────────
    ov_data = {}
    for nome_ov, (Xo, yo) in [('Treino',(X_tr,y_tr)),
                                ('Val',(X_val,y_val)),
                                ('Teste',(X_te,y_te))]:
        yp_ov  = model.predict(Xo)
        ypr_ov = model.predict_proba(Xo)[:,1]
        ov_data[nome_ov] = {
            'Acc':     accuracy_score(yo, yp_ov),
            'F1':      f1_score(yo, yp_ov, zero_division=0),
            'ROC-AUC': roc_auc_score(yo, ypr_ov),
        }
    df_ov = pd.DataFrame(ov_data).T
    gap_roc = ov_data['Treino']['ROC-AUC'] - ov_data['Teste']['ROC-AUC']

    # ── Passo 18 — Walk-Forward Validation ───────────────────────────────────
    wf_rows  = []
    wf_rets  = []
    n_splits = min(5, max(2, (n - t2) // 30))
    test_sz  = max(20, (n - t2) // n_splits)
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_sz)

    X_all_s = scaler.transform(X)
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_all_s), 1):
        if len(tr_idx) < MIN or len(te_idx) < 5:
            continue
        sc_wf   = RobustScaler()
        Xw_tr   = sc_wf.fit_transform(X[tr_idx])
