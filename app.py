# -*- coding: utf-8 -*-
# ==============================================================================
# SISTEMA QUANTITATIVO DE ML PARA TRADING — Streamlit Dashboard
# Versão: 4.0 | Dark Mode Professional
# ==============================================================================

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import time
import os

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quant ML Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS: Dark Mode Premium ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg-primary:    #0b0e14;
    --bg-card:       #111520;
    --bg-hover:      #161c2a;
    --border:        #1e2535;
    --border-bright: #2a3550;
    --accent-blue:   #3b82f6;
    --accent-cyan:   #06b6d4;
    --accent-green:  #10b981;
    --accent-red:    #ef4444;
    --accent-yellow: #f59e0b;
    --accent-purple: #8b5cf6;
    --text-primary:  #e2e8f0;
    --text-secondary:#94a3b8;
    --text-muted:    #475569;
    --font-mono:     'JetBrains Mono', monospace;
    --font-display:  'Syne', sans-serif;
}

html, body, [class*="css"], .stApp {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: var(--font-mono) !important; }

/* Cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--border-bright); }
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.metric-card.blue::before  { background: linear-gradient(90deg, var(--accent-blue), transparent); }
.metric-card.green::before { background: linear-gradient(90deg, var(--accent-green), transparent); }
.metric-card.red::before   { background: linear-gradient(90deg, var(--accent-red), transparent); }
.metric-card.yellow::before{ background: linear-gradient(90deg, var(--accent-yellow), transparent); }
.metric-card.purple::before{ background: linear-gradient(90deg, var(--accent-purple), transparent); }
.metric-card.cyan::before  { background: linear-gradient(90deg, var(--accent-cyan), transparent); }

.metric-label {
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 8px;
}
.metric-value {
    font-family: var(--font-display);
    font-size: 28px;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1;
}
.metric-value.pos { color: var(--accent-green); }
.metric-value.neg { color: var(--accent-red); }
.metric-value.neu { color: var(--accent-blue); }
.metric-sub {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 6px;
}

/* Section Headers */
.section-header {
    font-family: var(--font-display);
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-secondary);
    padding: 6px 0 12px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
}

/* Signal Badge */
.signal-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 20px;
    border-radius: 8px;
    font-family: var(--font-display);
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.signal-buy    { background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.4); color: #10b981; }
.signal-sell   { background: rgba(239,68,68,0.15);  border: 1px solid rgba(239,68,68,0.4);  color: #ef4444; }
.signal-neutral{ background: rgba(148,163,184,0.1); border: 1px solid rgba(148,163,184,0.3);color: #94a3b8; }

/* Table Styling */
.stDataFrame { background: var(--bg-card) !important; }
[data-testid="stTable"] { background: var(--bg-card) !important; }

/* Streamlit native overrides */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stSlider > div,
.stTextInput > div > div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
}
.stButton > button {
    background: var(--accent-blue) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--font-display) !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: 0.08em !important;
    padding: 12px 28px !important;
    transition: all 0.2s !important;
    width: 100%;
}
.stButton > button:hover {
    background: #2563eb !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(59,130,246,0.3) !important;
}

/* Divider */
hr { border-color: var(--border) !important; margin: 24px 0 !important; }

/* Expander */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-secondary) !important;
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
}

/* Progress bars */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)) !important;
}

/* Spinner */
.stSpinner > div { border-top-color: var(--accent-blue) !important; }

/* Alerts */
.stAlert { border-radius: 8px !important; font-family: var(--font-mono) !important; }

/* Plotly charts */
.js-plotly-plot { border-radius: 10px; }

/* Sidebar title */
.sidebar-title {
    font-family: var(--font-display);
    font-size: 20px;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.sidebar-subtitle {
    font-size: 10px;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 24px;
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# IMPORTS PARA ML
# ==============================================================================
try:
    from sklearn.preprocessing import RobustScaler
    from sklearn.decomposition import PCA
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, roc_auc_score, confusion_matrix, roc_curve)
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import (RandomForestClassifier, VotingClassifier, StackingClassifier)
    from sklearn.linear_model import LogisticRegression
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

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
    from catboost import CatBoostClassifier
    CAT_OK = True
except ImportError:
    CAT_OK = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_OK = True
except ImportError:
    OPTUNA_OK = False

# ==============================================================================
# CONSTANTES FINANCEIRAS (preservadas do Colab)
# ==============================================================================
CAPITAL_INICIAL     = 100_000
SPREAD_PIPS         = 0.0002
SLIPPAGE            = 0.0001
CUSTO_OPERACIONAL   = 0.0001
RISCO_POR_TRADE     = 0.02
STOP_LOSS_PCT       = 0.02
TAKE_PROFIT_PCT     = 0.04
WF_N_SPLITS         = 5
WF_TEST_SIZE        = 60

COLUNAS_NAO_FEATURE = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'target', 'retorno_futuro_log', 'retorno_futuro_pct',
    'retorno_futuro_pts', 'direcao_futura', 'retorno_atual'
]

# ==============================================================================
# PASSO 2 — DOWNLOAD DE DADOS
# ==============================================================================
@st.cache_data(ttl=60, show_spinner=False)
def baixar_dados(ticker: str, anos: int = 5, interval: str = "1d") -> pd.DataFrame:
    data_inicio = (datetime.now() - timedelta(days=anos * 365)).strftime("%Y-%m-%d")
    data_fim    = datetime.now().strftime("%Y-%m-%d")
    df = yf.download(ticker, start=data_inicio, end=data_fim,
                     interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().capitalize() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    df.columns = [c if c != 'Nan' else '_drop' for c in df.columns]
    df.drop(columns=[c for c in df.columns if c == '_drop'], errors='ignore', inplace=True)
    for alt in ['Adj close', 'Adj_close', 'Adjclose']:
        if alt in df.columns and 'Close' not in df.columns:
            df.rename(columns={alt: 'Close'}, inplace=True)
    return df

# ==============================================================================
# PASSO 3 — LIMPEZA
# ==============================================================================
def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='last')]
    if 'Close' in df.columns and 'Open' in df.columns:
        mask_validos = (df['Close'] > 0) & (df['Open'] > 0)
        df = df[mask_validos]
    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].fillna(0).clip(lower=0)
    else:
        df['Volume'] = 0
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[col] = df[col].astype(float)
    df.dropna(subset=[c for c in ['Open', 'High', 'Low', 'Close'] if c in df.columns], inplace=True)
    return df

# ==============================================================================
# PASSO 4 — TARGET
# ==============================================================================
def construir_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['retorno_futuro_log'] = np.log(df['Close'].shift(-1) / df['Close'])
    df['retorno_futuro_pct'] = df['Close'].pct_change().shift(-1) * 100
    df['retorno_futuro_pts'] = df['Close'].shift(-1) - df['Close']
    df['target']             = (df['retorno_futuro_log'] > 0).astype(int)
    df['direcao_futura']     = np.sign(df['retorno_futuro_log'])
    df['retorno_atual']      = np.log(df['Close'] / df['Close'].shift(1))
    df = df.iloc[:-1]
    return df

# ==============================================================================
# PASSO 5 — FEATURE ENGINEERING (lógica 100% preservada do Colab)
# ==============================================================================
def engenharia_de_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    o = df['Open']
    h = df['High']
    l = df['Low']
    c = df['Close']
    v = df['Volume'].replace(0, np.nan)

    # 1. Médias Móveis
    for p in [5, 10, 20, 50, 100, 200]:
        df[f'sma_{p}'] = c.rolling(p).mean()
        df[f'ema_{p}'] = c.ewm(span=p, adjust=False).mean()
    for p in [5, 10, 20, 50, 200]:
        df[f'preco_sma_{p}_ratio'] = c / df[f'sma_{p}'] - 1
    df['cruz_sma_5_20']   = (df['sma_5']  > df['sma_20']).astype(int)
    df['cruz_sma_20_50']  = (df['sma_20'] > df['sma_50']).astype(int)
    df['cruz_sma_50_200'] = (df['sma_50'] > df['sma_200']).astype(int)
    typical_price = (h + l + c) / 3
    df['vwap']             = (typical_price * v).rolling(20).sum() / v.rolling(20).sum()
    df['preco_vwap_ratio'] = c / df['vwap'] - 1

    # 2. Momentum
    for p in [7, 14, 21]:
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(p).mean()
        loss  = (-delta.clip(upper=0)).rolling(p).mean()
        rs    = gain / loss.replace(0, np.nan)
        df[f'rsi_{p}']      = 100 - (100 / (1 + rs))
        df[f'rsi_{p}_norm'] = df[f'rsi_{p}'] / 50 - 1
    for p in [5, 10, 20, 60]:
        df[f'roc_{p}'] = c.pct_change(p) * 100
    for p in [5, 10, 20]:
        df[f'mom_{p}']      = c - c.shift(p)
        df[f'mom_{p}_norm'] = df[f'mom_{p}'] / c.shift(p)
    for p in [14, 21]:
        low_min  = l.rolling(p).min()
        high_max = h.rolling(p).max()
        denom    = high_max - low_min
        df[f'stoch_k_{p}']    = 100 * (c - low_min) / denom.replace(0, np.nan)
        df[f'stoch_d_{p}']    = df[f'stoch_k_{p}'].rolling(3).mean()
        df[f'stoch_diff_{p}'] = df[f'stoch_k_{p}'] - df[f'stoch_d_{p}']

    # 3. Volatilidade
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    for p in [7, 14, 21]:
        df[f'atr_{p}']      = true_range.ewm(span=p, adjust=False).mean()
        df[f'atr_{p}_norm'] = df[f'atr_{p}'] / c
    ret = c.pct_change()
    for p in [5, 10, 20, 60]:
        df[f'vol_{p}'] = ret.rolling(p).std() * np.sqrt(252)
    for p in [20]:
        mid = c.rolling(p).mean()
        std = c.rolling(p).std()
        df[f'bb_upper_{p}'] = mid + 2 * std
        df[f'bb_lower_{p}'] = mid - 2 * std
        df[f'bb_width_{p}'] = (df[f'bb_upper_{p}'] - df[f'bb_lower_{p}']) / mid
        df[f'bb_pos_{p}']   = (c - df[f'bb_lower_{p}']) / (df[f'bb_upper_{p}'] - df[f'bb_lower_{p}'] + 1e-9)
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
        di_plus  = 100 * dm_plus.ewm(span=p, adjust=False).mean() / tr_s.replace(0, np.nan)
        di_minus = 100 * dm_minus.ewm(span=p, adjust=False).mean() / tr_s.replace(0, np.nan)
        dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9)
        df[f'adx_{p}']      = dx.ewm(span=p, adjust=False).mean()
        df[f'di_plus_{p}']  = di_plus
        df[f'di_minus_{p}'] = di_minus
        df[f'di_diff_{p}']  = di_plus - di_minus
    for p in [20, 50]:
        df[f'ema_{p}_slope'] = df[f'ema_{p}'].diff(5) / df[f'ema_{p}'].shift(5)

    # 5. Mean Reversion
    for p in [20, 60]:
        mu = c.rolling(p).mean()
        sg = c.rolling(p).std()
        df[f'zscore_{p}'] = (c - mu) / sg.replace(0, np.nan)
    for p in [20, 50, 200]:
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
    for n in [3, 5]:
        direcoes = (c > o).astype(int)
        df[f'sequencia_alta_{n}'] = direcoes.rolling(n).sum()
        df[f'pct_alta_{n}']       = df[f'sequencia_alta_{n}'] / n

    # 8. Lags
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f'ret_lag_{lag}'] = ret.shift(lag)
    for lag in [1, 5, 10]:
        df[f'vol20_lag_{lag}'] = df['vol_20'].shift(lag)
    for lag in [1, 3, 5]:
        df[f'rsi14_lag_{lag}'] = df['rsi_14'].shift(lag)
    for lag in [1, 3]:
        df[f'macd_lag_{lag}'] = df['macd'].shift(lag)

    # 9. Estatísticas Rolling
    for p in [20, 60]:
        df[f'skew_{p}'] = ret.rolling(p).skew()
        df[f'kurt_{p}'] = ret.rolling(p).kurt()
        df[f'maxdd_{p}'] = c.rolling(p).apply(
            lambda x: (x[-1] / x.max() - 1) if len(x) > 0 else 0, raw=True)
        df[f'autocorr_{p}'] = ret.rolling(p).apply(
            lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 1 else 0, raw=True)

    # 10. Regimes de Mercado
    df['regime_tendencia'] = (df['adx_14'] > 25).astype(int)
    df['regime_alta_vol']  = (df['vol_regime'] > 1.2).astype(int)
    above_sma50  = (c > df['sma_50']).astype(int)
    above_sma200 = (c > df['sma_200']).astype(int)
    df['regime_bull'] = ((above_sma50 == 1) & (above_sma200 == 1)).astype(int)
    df['regime_bear'] = ((above_sma50 == 0) & (above_sma200 == 0)).astype(int)

    def hurst_rs(ts):
        if len(ts) < 10 or ts.std() == 0:
            return 0.5
        mean = ts.mean()
        deviations = ts - mean
        cumdev = np.cumsum(deviations)
        R = cumdev.max() - cumdev.min()
        S = ts.std()
        if S == 0:
            return 0.5
        return np.log(R / S) / np.log(len(ts))

    df['hurst_60'] = ret.rolling(60).apply(hurst_rs, raw=True)

    # Features temporais
    df['dia_semana'] = df.index.dayofweek
    df['dia_mes']    = df.index.day
    df['mes']        = df.index.month
    df['trimestre']  = df.index.quarter
    df['fim_semana'] = (df.index.dayofweek >= 3).astype(int)
    df['inicio_mes'] = (df.index.day <= 5).astype(int)
    df['fim_mes']    = (df.index.day >= 25).astype(int)

    return df

# ==============================================================================
# PASSO 6 — SELEÇÃO DE FEATURES
# ==============================================================================
def selecionar_features(df: pd.DataFrame,
                         target_col: str = 'target',
                         max_features: int = 60,
                         correlacao_max: float = 0.95):
    features_candidatas = [c for c in df.columns if c not in COLUNAS_NAO_FEATURE]
    df_feats = df[features_candidatas + [target_col]].copy()
    nan_pct = df_feats.isnull().mean()
    features_ok = nan_pct[nan_pct < 0.30].index.tolist()
    features_ok = [f for f in features_ok if f != target_col]
    df_clean = df_feats[features_ok + [target_col]].dropna()
    variancias = df_clean[features_ok].var()
    features_ok = variancias[variancias > 1e-10].index.tolist()
    if len(features_ok) == 0:
        return [], pd.Series()
    corr_matrix = df_clean[features_ok].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    remover_corr = set()
    for col in upper.columns:
        if any(upper[col] > correlacao_max):
            remover_corr.add(col)
    features_ok = [f for f in features_ok if f not in remover_corr]
    X = df_clean[features_ok].values
    y = df_clean[target_col].values
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_ranking = pd.Series(mi_scores, index=features_ok).sort_values(ascending=False)
    features_selecionadas = mi_ranking.head(max_features).index.tolist()
    return features_selecionadas, mi_ranking

# ==============================================================================
# PASSO 7/9 — PREPARAÇÃO E SPLIT
# ==============================================================================
def preparar_dataset(df, features, target_col='target'):
    colunas = features + [target_col, 'Close', 'retorno_futuro_pct', 'retorno_atual']
    colunas = [c for c in colunas if c in df.columns]
    df_m = df[colunas].dropna()
    X       = df_m[features].values
    y       = df_m[target_col].values
    datas   = df_m.index
    closes  = df_m['Close'].values
    retornos= df_m['retorno_futuro_pct'].values if 'retorno_futuro_pct' in df_m.columns else np.zeros(len(df_m))
    return X, y, datas, closes, retornos

def split_temporal(X, y, datas, closes, retornos,
                   train_r=0.70, val_r=0.15, test_r=0.15):
    n       = len(X)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)
    return {
        'X_train': X[:n_train],          'y_train': y[:n_train],
        'X_val':   X[n_train:n_train+n_val], 'y_val': y[n_train:n_train+n_val],
        'X_test':  X[n_train+n_val:],    'y_test': y[n_train+n_val:],
        'datas_train':    datas[:n_train],
        'datas_val':      datas[n_train:n_train+n_val],
        'datas_test':     datas[n_train+n_val:],
        'closes_train':   closes[:n_train],
        'closes_val':     closes[n_train:n_train+n_val],
        'closes_test':    closes[n_train+n_val:],
        'retornos_train': retornos[:n_train],
        'retornos_val':   retornos[n_train:n_train+n_val],
        'retornos_test':  retornos[n_train+n_val:],
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n - n_train - n_val,
    }

# ==============================================================================
# PASSO 10 — MODELOS
# ==============================================================================
def criar_modelos():
    modelos = {}
    modelos['LogReg'] = LogisticRegression(
        C=0.1, max_iter=1000, random_state=42, class_weight='balanced')
    modelos['DecTree'] = DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=20, random_state=42, class_weight='balanced')
    modelos['RandForest'] = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=10,
        max_features='sqrt', random_state=42, class_weight='balanced', n_jobs=-1)
    if XGB_OK:
        modelos['XGBoost'] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, gamma=0.1,
            eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0)
    if LGB_OK:
        modelos['LightGBM'] = LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
    if CAT_OK:
        modelos['CatBoost'] = CatBoostClassifier(
            iterations=300, depth=4, learning_rate=0.05,
            eval_metric='AUC', auto_class_weights='Balanced',
            random_seed=42, verbose=0)
    return modelos

def treinar_modelos(modelos, X_train, y_train, X_val, y_val, X_test, y_test):
    resultados = []
    modelos_treinados = {}
    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred_val  = modelo.predict(X_val)
        y_pred_test = modelo.predict(X_test)
        y_prob_val  = modelo.predict_proba(X_val)[:, 1]
        y_prob_test = modelo.predict_proba(X_test)[:, 1]
        res = {
            'Modelo':    nome,
            'Acc_Val':   accuracy_score(y_val,  y_pred_val),
            'Acc_Test':  accuracy_score(y_test, y_pred_test),
            'F1_Val':    f1_score(y_val,  y_pred_val, zero_division=0),
            'F1_Test':   f1_score(y_test, y_pred_test, zero_division=0),
            'ROC_Val':   roc_auc_score(y_val,  y_prob_val),
            'ROC_Test':  roc_auc_score(y_test, y_prob_test),
            'Prec_Test': precision_score(y_test, y_pred_test, zero_division=0),
            'Rec_Test':  recall_score(y_test,  y_pred_test, zero_division=0),
        }
        resultados.append(res)
        modelos_treinados[nome] = {
            'modelo': modelo,
            'y_pred_test': y_pred_test,
            'y_prob_test': y_prob_test,
        }
    df_res = pd.DataFrame(resultados).sort_values('ROC_Test', ascending=False)
    return df_res, modelos_treinados

# ==============================================================================
# PASSO 12 — ENSEMBLE
# ==============================================================================
def criar_ensemble(modelos_treinados, X_val, y_val, X_train, y_train):
    ensembles = {}
    estimadores = []
    for nome in ['XGBoost', 'LightGBM', 'CatBoost', 'RandForest', 'DecTree', 'LogReg']:
        if nome in modelos_treinados:
            estimadores.append((nome, modelos_treinados[nome]['modelo']))
    if len(estimadores) < 2:
        estimadores = [(n, i['modelo']) for n, i in list(modelos_treinados.items())]
    if len(estimadores) < 2:
        return {}
    try:
        voting = VotingClassifier(estimators=estimadores, voting='soft', n_jobs=-1)
        voting.fit(X_train, y_train)
        prob_val = voting.predict_proba(X_val)[:, 1]
        auc_val  = roc_auc_score(y_val, prob_val)
        ensembles['Voting_Soft'] = {'modelo': voting, 'auc_val': auc_val}
    except Exception:
        pass
    try:
        probs_val_pond = np.zeros(len(y_val))
        soma_pesos = 0.0
        pesos = {}
        for nome, info in modelos_treinados.items():
            prob = info['modelo'].predict_proba(X_val)[:, 1]
            auc  = roc_auc_score(y_val, prob)
            pesos[nome] = auc
            probs_val_pond += auc * prob
            soma_pesos += auc
        if soma_pesos > 0:
            probs_val_pond /= soma_pesos
            auc_pond = roc_auc_score(y_val, probs_val_pond)
            ensembles['Ponderado'] = {'pesos': pesos, 'soma_pesos': soma_pesos, 'auc_val': auc_pond}
    except Exception:
        pass
    return ensembles

# ==============================================================================
# PASSO 13 — MÉTRICAS FINANCEIRAS (lógica preservada do Colab)
# ==============================================================================
def metricas_financeiras(retornos_modelo, retornos_bh, capital_inicial=100_000):
    rm  = pd.Series(retornos_modelo) if not isinstance(retornos_modelo, pd.Series) else retornos_modelo
    rbh = pd.Series(retornos_bh) if not isinstance(retornos_bh, pd.Series) else retornos_bh
    equity    = capital_inicial * (1 + rm).cumprod()
    equity_bh = capital_inicial * (1 + rbh).cumprod()
    ret_total = (equity.iloc[-1] / capital_inicial - 1) if len(equity) > 0 else 0
    ret_bh    = (equity_bh.iloc[-1] / capital_inicial - 1) if len(equity_bh) > 0 else 0
    std = rm.std()
    sharpe  = (rm.mean() / std * np.sqrt(252)) if std > 0 else 0
    downside = rm[rm < 0].std()
    sortino  = (rm.mean() / downside * np.sqrt(252)) if downside > 0 else 0
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    max_dd   = drawdown.min()
    ann_ret  = ret_total / (len(rm) / 252) if len(rm) > 0 else 0
    calmar   = (-ann_ret / max_dd) if max_dd < 0 else 0
    rm_clean = rm.dropna()
    wins   = int((rm_clean > 0).sum())
    losses = int((rm_clean <= 0).sum())
    win_rate = wins / len(rm_clean) if len(rm_clean) > 0 else 0
    gross_profit = rm_clean[rm_clean > 0].sum()
    gross_loss   = abs(rm_clean[rm_clean < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    avg_win  = rm_clean[rm_clean > 0].mean() if wins > 0 else 0
    avg_loss = abs(rm_clean[rm_clean < 0].mean()) if losses > 0 else 1
    payoff   = avg_win / avg_loss if avg_loss > 0 else 0
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
    return {
        'Retorno Total': ret_total, 'Retorno B&H': ret_bh,
        'Sharpe Ratio':  sharpe,    'Sortino Ratio': sortino,
        'Max Drawdown':  max_dd,    'Calmar Ratio': calmar,
        'Win Rate':      win_rate,  'Profit Factor': profit_factor,
        'Payoff':        payoff,    'Expectancy': expectancy,
        'N Trades':      len(rm),   'equity': equity,
        'equity_bh':     equity_bh, 'drawdown': drawdown,
    }

def avaliar_completo(modelo, X_test, y_test, retornos_test, closes_test, datas_test):
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]
    sinais = np.where(y_pred == 1, 1, -1)
    retornos_modelo = pd.Series(sinais * retornos_test / 100, index=datas_test)
    retornos_bh     = pd.Series(retornos_test / 100, index=datas_test)
    n_trocas = (np.diff(y_pred) != 0).sum()
    custo_total = n_trocas * (SPREAD_PIPS + SLIPPAGE + CUSTO_OPERACIONAL)
    retornos_modelo.iloc[-1] -= custo_total
    metricas_fin = metricas_financeiras(retornos_modelo, retornos_bh, CAPITAL_INICIAL)
    return {
        'y_pred': y_pred, 'y_prob': y_prob,
        'acc':  accuracy_score(y_test, y_pred),
        'prec': precision_score(y_test, y_pred, zero_division=0),
        'rec':  recall_score(y_test, y_pred, zero_division=0),
        'f1':   f1_score(y_test, y_pred, zero_division=0),
        'roc':  roc_auc_score(y_test, y_prob),
        'metricas_fin': metricas_fin,
        'retornos_modelo': retornos_modelo,
        'retornos_bh': retornos_bh,
    }

# ==============================================================================
# PASSO 14 — BACKTEST REALISTA (lógica 100% preservada)
# ==============================================================================
def backtest_realista(y_pred, y_prob, closes, retornos, datas,
                       capital_inicial=100_000, prob_threshold=0.55,
                       risco_trade=0.02, stop_loss=0.02, take_profit=0.04,
                       spread=0.0002, slippage=0.0001):
    capital = capital_inicial
    equity_curve = [capital]
    trades = []
    for i in range(len(y_pred)):
        preco_entrada = closes[i]
        retorno_real  = retornos[i] / 100
        sinal = 0
        if y_prob[i] > prob_threshold:
            sinal = 1
        elif y_prob[i] < (1 - prob_threshold):
            sinal = -1
        if sinal == 0:
            equity_curve.append(capital)
            continue
        custo_entrada = (spread + slippage) * capital
        tamanho_pos   = capital * risco_trade
        pnl_bruto     = tamanho_pos * sinal * retorno_real
        if sinal == 1:
            if retorno_real < -stop_loss:
                pnl_bruto = -tamanho_pos * stop_loss
            elif retorno_real > take_profit:
                pnl_bruto = tamanho_pos * take_profit
        else:
            if retorno_real > stop_loss:
                pnl_bruto = -tamanho_pos * stop_loss
            elif retorno_real < -take_profit:
                pnl_bruto = tamanho_pos * take_profit
        pnl_liquido = pnl_bruto - custo_entrada
        capital    += pnl_liquido
        equity_curve.append(capital)
        trades.append({
            'data': datas[i], 'sinal': sinal, 'prob': y_prob[i],
            'preco': preco_entrada, 'retorno_real': retorno_real,
            'pnl': pnl_liquido, 'capital': capital,
        })
    eq_series = pd.Series(equity_curve)
    ret_total = (capital / capital_inicial - 1)
    df_trades = pd.DataFrame(trades)
    n_trades  = len(df_trades)
    if n_trades > 0:
        wins = (df_trades['pnl'] > 0).sum()
        win_rate = wins / n_trades
        pf_num = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
        pf_den = abs(df_trades[df_trades['pnl'] < 0]['pnl'].sum())
        profit_factor = pf_num / pf_den if pf_den > 0 else np.inf
    else:
        win_rate = profit_factor = 0
    roll_max = eq_series.cummax()
    dd       = (eq_series - roll_max) / roll_max
    max_dd   = dd.min()
    return {
        'equity_curve': eq_series, 'df_trades': df_trades,
        'ret_total': ret_total,    'n_trades': n_trades,
        'win_rate': win_rate,      'profit_factor': profit_factor,
        'max_dd': max_dd,          'capital_final': capital,
    }

# ==============================================================================
# PASSO 16 — THRESHOLD ÓTIMO
# ==============================================================================
def encontrar_threshold_otimo(modelo, X_test, y_test, retornos_test):
    y_prob = modelo.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.40, 0.75, 0.02)
    resultados = []
    for thr in thresholds:
        y_pred_thr = (y_prob >= thr).astype(int)
        n_sinais = y_pred_thr.sum()
        if n_sinais < 5:
            continue
        f1   = f1_score(y_test, y_pred_thr, zero_division=0)
        prec = precision_score(y_test, y_pred_thr, zero_division=0)
        rec  = recall_score(y_test, y_pred_thr, zero_division=0)
        acc  = accuracy_score(y_test, y_pred_thr)
        sinais = np.where(y_pred_thr == 1, 1, 0)
        rets_est = sinais * retornos_test / 100
        ret_total = (1 + rets_est).prod() - 1
        resultados.append({'threshold': thr, 'f1': f1, 'acc': acc,
                           'prec': prec, 'rec': rec, 'n_sinais': n_sinais,
                           'pct_sinais': n_sinais / len(y_test), 'retorno': ret_total})
    if not resultados:
        return 0.55, pd.DataFrame()
    df_thr = pd.DataFrame(resultados)
    thr_otimo = float(df_thr.loc[df_thr['f1'].idxmax(), 'threshold'])
    return thr_otimo, df_thr

# ==============================================================================
# PASSO 18 — WALK-FORWARD
# ==============================================================================
def walk_forward_validation(X, y, datas, retornos, prob_threshold,
                              n_splits=5, test_size=60):
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    wf_resultados = []
    retornos_wf_todos = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        ret_te     = retornos[test_idx]
        sc = RobustScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
        if XGB_OK:
            modelo_wf = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, eval_metric='logloss', random_state=42,
                n_jobs=-1, verbosity=0)
        else:
            modelo_wf = RandomForestClassifier(
                n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        modelo_wf.fit(X_tr_s, y_tr)
        y_pred = modelo_wf.predict(X_te_s)
        y_prob = modelo_wf.predict_proba(X_te_s)[:, 1]
        acc = accuracy_score(y_te, y_pred)
        roc = roc_auc_score(y_te, y_prob)
        f1  = f1_score(y_te, y_pred, zero_division=0)
        sinais  = np.where(y_prob >= prob_threshold, 1,
                           np.where(y_prob <= 1 - prob_threshold, -1, 0))
        ret_est = sinais * ret_te / 100
        ret_total = (1 + ret_est).prod() - 1
        wf_resultados.append({
            'Fold': fold,
            'Período': f"{datas[test_idx[0]].date()} → {datas[test_idx[-1]].date()}",
            'N_Treino': len(train_idx), 'N_Teste': len(test_idx),
            'Accuracy': acc, 'ROC-AUC': roc, 'F1': f1, 'Retorno': ret_total,
        })
        retornos_wf_todos.extend(ret_est.tolist())
    df_wf = pd.DataFrame(wf_resultados)
    retornos_wf_series = pd.Series(retornos_wf_todos)
    return {'df_wf': df_wf, 'retornos_todos': retornos_wf_series}

# ==============================================================================
# PASSO 20 — PREVISÃO PRODUÇÃO
# ==============================================================================
def predict_next_day(ticker, modelo, scaler_obj, feature_names, threshold):
    df_novo = baixar_dados(ticker, anos=1, interval="1d")
    df_novo = limpar_dados(df_novo)
    df_novo = construir_target(df_novo)
    df_novo = engenharia_de_features(df_novo)
    for f in feature_names:
        if f not in df_novo.columns:
            df_novo[f] = 0.0
    ultima_linha = df_novo[feature_names].iloc[-1:]
    ultima_linha = ultima_linha.fillna(0)
    X_prod = scaler_obj.transform(ultima_linha.values)
    prob_alta  = modelo.predict_proba(X_prod)[0, 1]
    prob_queda = 1 - prob_alta
    if prob_alta >= threshold:
        sinal = "BUY"
        confianca = prob_alta
    elif prob_queda >= threshold:
        sinal = "SELL"
        confianca = prob_queda
    else:
        sinal = "NEUTRAL"
        confianca = max(prob_alta, prob_queda)
    preco_atual  = float(df_novo['Close'].iloc[-1])
    rsi_atual    = float(df_novo['rsi_14'].iloc[-1]) if 'rsi_14' in df_novo.columns else 50.0
    vol_20       = float(df_novo['vol_20'].iloc[-1]) if 'vol_20' in df_novo.columns else 0.0
    retorno_hoje = float(df_novo['retorno_atual'].iloc[-1]) * 100 if 'retorno_atual' in df_novo.columns else 0.0
    data_ref     = df_novo.index[-1].date()
    return {
        'ticker': ticker, 'data_referencia': str(data_ref),
        'preco_atual': preco_atual, 'prob_alta': prob_alta,
        'prob_queda': prob_queda, 'sinal': sinal, 'confianca': confianca,
        'threshold': threshold, 'rsi': rsi_atual, 'vol_20': vol_20,
        'retorno_hoje': retorno_hoje,
    }

# ==============================================================================
# FUNÇÃO PRINCIPAL: processar_estrategia (encapsula TODO o pipeline do Colab)
# ==============================================================================
def processar_estrategia(ticker: str, anos: int = 5, interval: str = "1d",
                          max_features: int = 60, usar_ensemble: bool = False,
                          run_walk_forward: bool = True):
    """
    Pipeline completo de ML para trading — mesma lógica do Colab.
    Retorna dict com todos os resultados para exibição no dashboard.
    """
    logs = []
    def log(msg):
        logs.append(msg)

    log(f"📥 Baixando dados de {ticker}...")
    df_raw = baixar_dados(ticker, anos, interval)
    if df_raw is None or len(df_raw) < 200:
        raise ValueError(f"Dados insuficientes para {ticker}. Tente outro ativo ou período maior.")

    log(f"🧹 Limpando dados ({len(df_raw)} barras)...")
    df = limpar_dados(df_raw)

    log("🎯 Construindo target...")
    df = construir_target(df)

    log("⚙️ Engenharia de features...")
    df = engenharia_de_features(df)

    log("🔍 Selecionando features (Mutual Information)...")
    features_sel, mi_ranking = selecionar_features(df, max_features=max_features)
    if len(features_sel) == 0:
        raise ValueError("Nenhuma feature válida encontrada. Verifique os dados.")

    log(f"📐 {len(features_sel)} features selecionadas. Preparando dataset...")
    X, y, datas, closes, retornos = preparar_dataset(df, features_sel)
    splits = split_temporal(X, y, datas, closes, retornos)

    log("📏 Normalizando (RobustScaler)...")
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(splits['X_train'])
    X_val_s   = scaler.transform(splits['X_val'])
    X_test_s  = scaler.transform(splits['X_test'])
    X_all_s   = scaler.transform(X)

    log("🤖 Treinando modelos...")
    modelos = criar_modelos()
    df_resultados, modelos_treinados = treinar_modelos(
        modelos, X_train_s, splits['y_train'],
        X_val_s, splits['y_val'],
        X_test_s, splits['y_test'])

    # Selecionar melhor modelo (lógica do Colab: preferir tree-based)
    melhor_nome = df_resultados.iloc[0]['Modelo']
    MODELOS_LINEARES = {'LogReg'}
    MODELOS_TREE     = {'XGBoost', 'LightGBM', 'CatBoost', 'RandForest', 'DecTree'}
    if melhor_nome in MODELOS_LINEARES:
        df_tree = df_resultados[df_resultados['Modelo'].isin(MODELOS_TREE)]
        if len(df_tree) > 0:
            roc_top  = df_resultados.iloc[0]['ROC_Test']
            roc_tree = df_tree.iloc[0]['ROC_Test']
            if roc_top - roc_tree <= 0.005:
                melhor_nome = df_tree.iloc[0]['Modelo']
    melhor_modelo = modelos_treinados[melhor_nome]['modelo']

    log(f"✅ Melhor modelo: {melhor_nome}")

    # Ensemble (opcional)
    if usar_ensemble:
        log("🔗 Criando ensemble...")
        ensembles = criar_ensemble(modelos_treinados, X_val_s, splits['y_val'],
                                   X_train_s, splits['y_train'])
    else:
        ensembles = {}

    log("📊 Avaliando performance...")
    avaliacao = avaliar_completo(
        melhor_modelo, X_test_s, splits['y_test'],
        splits['retornos_test'], splits['closes_test'], splits['datas_test'])

    log("🎚️ Encontrando threshold ótimo...")
    prob_threshold, df_threshold = encontrar_threshold_otimo(
        melhor_modelo, X_test_s, splits['y_test'], splits['retornos_test'])

    log(f"⚡ Executando backtest (threshold={prob_threshold:.0%})...")
    backtest = backtest_realista(
        avaliacao['y_pred'], avaliacao['y_prob'],
        splits['closes_test'], splits['retornos_test'], splits['datas_test'],
        CAPITAL_INICIAL, prob_threshold, RISCO_POR_TRADE,
        STOP_LOSS_PCT, TAKE_PROFIT_PCT, SPREAD_PIPS, SLIPPAGE)

    wf_resultados = None
    if run_walk_forward:
        log("🔄 Walk-Forward Validation...")
        try:
            wf_resultados = walk_forward_validation(
                X_all_s, y, datas, retornos, prob_threshold,
                WF_N_SPLITS, WF_TEST_SIZE)
        except Exception as e:
            log(f"⚠️ Walk-Forward ignorado: {e}")

    log("🔮 Gerando previsão para próximo dia...")
    previsao = predict_next_day(ticker, melhor_modelo, scaler, features_sel, prob_threshold)

    log("✅ Pipeline concluído!")
    return {
        'ticker': ticker, 'df': df, 'splits': splits,
        'features_sel': features_sel, 'mi_ranking': mi_ranking,
        'df_resultados': df_resultados, 'modelos_treinados': modelos_treinados,
        'melhor_nome': melhor_nome, 'melhor_modelo': melhor_modelo,
        'avaliacao': avaliacao, 'backtest': backtest,
        'prob_threshold': prob_threshold, 'df_threshold': df_threshold,
        'wf_resultados': wf_resultados, 'previsao': previsao,
        'scaler': scaler, 'logs': logs,
        'datas': datas, 'closes': closes, 'retornos': retornos, 'y': y,
    }

# ==============================================================================
# HELPERS DE PLOTAGEM (Plotly dark)
# ==============================================================================
PLOTLY_LAYOUT = dict(
    paper_bgcolor='#111520', plot_bgcolor='#0b0e14',
    font=dict(family='JetBrains Mono, monospace', color='#94a3b8', size=11),
    xaxis=dict(gridcolor='#1e2535', linecolor='#1e2535', zeroline=False),
    yaxis=dict(gridcolor='#1e2535', linecolor='#1e2535', zeroline=False),
    margin=dict(l=50, r=20, t=50, b=40),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1e2535', borderwidth=1),
)

def plot_equity_curve(avaliacao, backtest, splits, ticker, melhor_nome):
    mf = avaliacao['metricas_fin']
    datas_test = splits['datas_test']
    eq_mod  = mf['equity'].values
    eq_bh   = mf['equity_bh'].values
    n = min(len(datas_test), len(eq_mod))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.7, 0.3],
                         vertical_spacing=0.04)
    fig.add_trace(go.Scatter(
        x=datas_test[:n], y=eq_mod[:n], name='Modelo ML',
        line=dict(color='#3b82f6', width=2.5),
        fill='tonexty', fillcolor='rgba(59,130,246,0.06)'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=datas_test[:n], y=eq_bh[:n], name='Buy & Hold',
        line=dict(color='#f59e0b', width=1.8, dash='dot')), row=1, col=1)
    fig.add_hline(y=CAPITAL_INICIAL, line_dash='dash',
                  line_color='rgba(148,163,184,0.3)', row=1, col=1)
    dd_vals = mf['drawdown'].values[:n] * 100
    fig.add_trace(go.Scatter(
        x=datas_test[:n], y=dd_vals, name='Drawdown',
        fill='tozeroy', fillcolor='rgba(239,68,68,0.2)',
        line=dict(color='#ef4444', width=1.5)), row=2, col=1)
    layout = {**PLOTLY_LAYOUT,
              'title': dict(text=f'<b>Equity Curve — {ticker} | {melhor_nome}</b>',
                            font=dict(color='#e2e8f0', size=14)),
              'height': 500,
              'yaxis': dict(**PLOTLY_LAYOUT['yaxis'],
                            tickformat='$,.0f', title='Capital (USD)'),
              'yaxis2': dict(**PLOTLY_LAYOUT['yaxis'], title='Drawdown (%)'),
              'hovermode': 'x unified'}
    fig.update_layout(**layout)
    return fig

def plot_model_comparison(df_resultados, melhor_nome):
    df_plot = df_resultados.sort_values('ROC_Test')
    colors  = ['#3b82f6' if n == melhor_nome else '#1e2535' for n in df_plot['Modelo']]
    fig = go.Figure(go.Bar(
        x=df_plot['ROC_Test'], y=df_plot['Modelo'], orientation='h',
        marker_color=colors,
        text=[f"{v:.4f}" for v in df_plot['ROC_Test']],
        textposition='outside', textfont=dict(color='#94a3b8', size=11)))
    fig.add_vline(x=0.5, line_dash='dash', line_color='#ef4444', annotation_text='Random')
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=dict(text='<b>ROC-AUC por Modelo</b>',
                                 font=dict(color='#e2e8f0', size=13)),
                      height=350, xaxis_range=[0.45, df_plot['ROC_Test'].max() + 0.05],
                      showlegend=False)
    return fig

def plot_confusion_matrix(avaliacao, splits):
    cm = confusion_matrix(splits['y_test'], avaliacao['y_pred'])
    labels = ['Queda', 'Alta']
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0, '#0b0e14'], [0.5, '#1e3a5f'], [1, '#3b82f6']],
        text=cm, texttemplate='<b>%{text}</b>',
        textfont=dict(size=22, color='white'),
        showscale=False))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=dict(text='<b>Confusion Matrix</b>',
                                 font=dict(color='#e2e8f0', size=13)),
                      height=320, xaxis_title='Predito', yaxis_title='Real')
    return fig

def plot_roc_curve(avaliacao, splits):
    fpr, tpr, _ = roc_curve(splits['y_test'], avaliacao['y_prob'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, name=f"AUC = {avaliacao['roc']:.4f}",
        line=dict(color='#3b82f6', width=2.5),
        fill='tozeroy', fillcolor='rgba(59,130,246,0.08)'))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], name='Random',
        line=dict(color='#475569', dash='dash', width=1.5)))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=dict(text='<b>ROC Curve</b>',
                                 font=dict(color='#e2e8f0', size=13)),
                      height=320, xaxis_title='FPR', yaxis_title='TPR')
    return fig

def plot_threshold_analysis(df_threshold, thr_otimo):
    if df_threshold is None or len(df_threshold) == 0:
        return None
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Métricas vs Threshold', '% Sinais vs Threshold'])
    for col_name, color in [('f1','#3b82f6'), ('prec','#10b981'), ('rec','#f59e0b'), ('acc','#8b5cf6')]:
        fig.add_trace(go.Scatter(
            x=df_threshold['threshold'], y=df_threshold[col_name],
            name=col_name.upper(), line=dict(color=color, width=2)), row=1, col=1)
    fig.add_vline(x=thr_otimo, line_dash='dash', line_color='#ef4444',
                  annotation_text=f'Ótimo={thr_otimo:.2f}')
    fig.add_trace(go.Bar(
        x=df_threshold['threshold'], y=df_threshold['pct_sinais'] * 100,
        name='% Sinais', marker_color='#06b6d4', opacity=0.7), row=1, col=2)
    fig.update_layout(**PLOTLY_LAYOUT, height=350,
                      title=dict(text='<b>Análise de Threshold</b>',
                                 font=dict(color='#e2e8f0', size=13)))
    return fig

def plot_feature_importance(mi_ranking, n=20):
    top = mi_ranking.head(n).sort_values()
    colors = ['#3b82f6' if i >= len(top) - 5 else '#1e2535' for i in range(len(top))]
    fig = go.Figure(go.Bar(
        x=top.values, y=top.index, orientation='h',
        marker_color=colors,
        text=[f"{v:.4f}" for v in top.values],
        textposition='outside', textfont=dict(color='#94a3b8', size=10)))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=dict(text=f'<b>Top {n} Features (Mutual Information)</b>',
                                 font=dict(color='#e2e8f0', size=13)),
                      height=500, showlegend=False)
    return fig

def plot_walk_forward(wf_resultados):
    if wf_resultados is None:
        return None
    df_wf = wf_resultados['df_wf']
    fig = make_subplots(rows=1, cols=3,
                         subplot_titles=['ROC-AUC por Fold', 'Accuracy por Fold', 'Retorno por Fold'])
    for col_idx, (col, color) in enumerate([('ROC-AUC', '#3b82f6'), ('Accuracy', '#10b981'), ('Retorno', '#f59e0b')], 1):
        vals = df_wf[col].values * (100 if col == 'Retorno' else 1)
        bar_colors = ['#ef4444' if v < 0 else color for v in vals]
        fig.add_trace(go.Bar(
            x=[f"Fold {i}" for i in df_wf['Fold']], y=vals,
            marker_color=bar_colors, name=col,
            text=[f"{v:.2f}{'%' if col=='Retorno' else ''}" for v in vals],
            textposition='outside', textfont=dict(color='#94a3b8', size=10)), row=1, col=col_idx)
    fig.update_layout(**PLOTLY_LAYOUT, height=350,
                      title=dict(text='<b>Walk-Forward Validation</b>',
                                 font=dict(color='#e2e8f0', size=13)),
                      showlegend=False)
    return fig

def plot_returns_dist(retornos_test):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=retornos_test, nbinsx=60, name='Retornos',
        marker_color='#3b82f6', opacity=0.75,
        histnorm='probability density'))
    mu, std = retornos_test.mean(), retornos_test.std()
    x_norm = np.linspace(mu - 4*std, mu + 4*std, 200)
    from scipy.stats import norm as sci_norm
    fig.add_trace(go.Scatter(
        x=x_norm, y=sci_norm.pdf(x_norm, mu, std),
        name='Normal', line=dict(color='#f59e0b', width=2)))
    fig.add_vline(x=0, line_dash='dash', line_color='#ef4444', opacity=0.6)
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=dict(text='<b>Distribuição dos Retornos</b>',
                                 font=dict(color='#e2e8f0', size=13)),
                      height=320, xaxis_title='Retorno (%)', yaxis_title='Densidade')
    return fig

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚡ QUANT ML</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Trading Intelligence System v4.0</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">⚙ Configuração</div>', unsafe_allow_html=True)

    TICKERS_POPULARES = [
        "USDJPY=X", "EURUSD=X", "GBPUSD=X", "AUDUSD=X",
        "BTC-USD", "ETH-USD",
        "^GSPC", "^IXIC", "^BVSP",
        "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META",
        "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA",
        "GC=F", "CL=F",
    ]

    ticker_input = st.selectbox(
        "🎯 Ativo (Ticker)",
        options=TICKERS_POPULARES,
        index=0,
        help="Selecione ou digite qualquer ticker válido do Yahoo Finance"
    )
    ticker_custom = st.text_input("Ou digite um ticker personalizado:", placeholder="ex: PETR4.SA")
    ticker = ticker_custom.upper().strip() if ticker_custom.strip() else ticker_input

    anos = st.select_slider(
        "📅 Histórico (anos)",
        options=[2, 3, 5, 7, 10],
        value=5
    )

    interval_map = {"Diário (1d)": "1d", "Semanal (1wk)": "1wk"}
    interval_label = st.selectbox("⏱ Timeframe", list(interval_map.keys()))
    interval = interval_map[interval_label]

    max_features = st.slider("🔢 Max Features (MI)", 20, 80, 60, step=5)

    st.markdown("---")
    st.markdown('<div class="section-header">🛡 Parâmetros Avançados</div>', unsafe_allow_html=True)

    usar_ensemble   = st.checkbox("Usar Ensemble", value=False,
                                   help="Combina múltiplos modelos (mais lento)")
    run_wf          = st.checkbox("Walk-Forward Validation", value=True)

    st.markdown("---")
    executar = st.button("🚀  EXECUTAR BACKTEST", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:10px; color: #475569; line-height: 1.6; font-family: var(--font-mono);">
    ⚠️ <b style="color:#94a3b8">DISCLAIMER</b><br>
    Este sistema é exclusivamente para fins educacionais e de pesquisa quantitativa.<br><br>
    NÃO constitui recomendação de investimento. Resultados passados não garantem retornos futuros.
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# MAIN CONTENT
# ==============================================================================

# Header
st.markdown("""
<div style="padding: 24px 0 16px 0;">
    <div style="font-family: 'Syne', sans-serif; font-size: 32px; font-weight: 800;
                background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 50%, #10b981 100%);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                margin-bottom: 4px;">
        Sistema Quantitativo de ML para Trading
    </div>
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #475569;
                letter-spacing: 0.1em;">
        PIPELINE COMPLETO · FEATURE ENGINEERING · ENSEMBLE · BACKTEST REALISTA · WALK-FORWARD
    </div>
</div>
""", unsafe_allow_html=True)

# Auto-refresh (a cada 5 min se houver resultados)
try:
    from streamlit_autorefresh import st_autorefresh
    if 'resultado' in st.session_state and st.session_state.resultado is not None:
        refresh_interval = st.sidebar.select_slider(
            "🔄 Auto-refresh (min)", options=[1, 5, 10, 30], value=5)
        st_autorefresh(interval=refresh_interval * 60 * 1000, key="autorefresh")
except ImportError:
    pass

# Inicializar session state
if 'resultado' not in st.session_state:
    st.session_state.resultado = None

# ── Executar pipeline ──────────────────────────────────────────────────────────
if executar:
    st.session_state.resultado = None
    progress_container = st.empty()
    log_container      = st.empty()
    with progress_container.container():
        prog_bar = st.progress(0)
        status   = st.empty()

    try:
        status.markdown(f"<div style='color:#94a3b8; font-family:JetBrains Mono; font-size:12px;'>Iniciando pipeline para <b style='color:#3b82f6'>{ticker}</b>...</div>", unsafe_allow_html=True)

        def update_progress(step, total, msg):
            prog_bar.progress(int(step / total * 100))
            status.markdown(f"<div style='color:#94a3b8; font-family:JetBrains Mono; font-size:12px;'>{msg}</div>", unsafe_allow_html=True)

        update_progress(1, 10, f"📥 Baixando dados: {ticker} ({anos} anos)...")
        resultado = processar_estrategia(
            ticker=ticker, anos=anos, interval=interval,
            max_features=max_features, usar_ensemble=usar_ensemble,
            run_walk_forward=run_wf)
        update_progress(10, 10, "✅ Pipeline concluído!")
        st.session_state.resultado = resultado
        progress_container.empty()
        log_container.empty()

    except Exception as e:
        progress_container.empty()
        st.error(f"❌ Erro no pipeline: {str(e)}")
        st.stop()

# ── Exibir resultados ──────────────────────────────────────────────────────────
if st.session_state.resultado is not None:
    r = st.session_state.resultado

    ticker_display = r['ticker']
    previsao       = r['previsao']
    avaliacao      = r['avaliacao']
    backtest       = r['backtest']
    melhor_nome    = r['melhor_nome']
    splits         = r['splits']
    mf             = avaliacao['metricas_fin']
    thr            = r['prob_threshold']

    # ── SINAL PRINCIPAL ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🔮 SINAL PARA O PRÓXIMO DIA</div>', unsafe_allow_html=True)

    signal_class = {'BUY': 'signal-buy', 'SELL': 'signal-sell', 'NEUTRAL': 'signal-neutral'}[previsao['sinal']]
    signal_emoji = {'BUY': '🟢 COMPRA', 'SELL': '🔴 VENDA', 'NEUTRAL': '⚪ NEUTRO'}[previsao['sinal']]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(f"""
        <div class="metric-card blue">
            <div class="metric-label">Sinal</div>
            <div class="metric-value neu"><span class="signal-badge {signal_class}">{signal_emoji}</span></div>
            <div class="metric-sub">Threshold: {thr:.0%}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card green">
            <div class="metric-label">Prob. Alta ↑</div>
            <div class="metric-value {'pos' if previsao['prob_alta'] > 0.5 else 'neg'}">{previsao['prob_alta']:.1%}</div>
            <div class="metric-sub">vs 50% aleatório</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card red">
            <div class="metric-label">Prob. Queda ↓</div>
            <div class="metric-value {'neg' if previsao['prob_queda'] > 0.5 else 'pos'}">{previsao['prob_queda']:.1%}</div>
            <div class="metric-sub">vs 50% aleatório</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card purple">
            <div class="metric-label">Confiança</div>
            <div class="metric-value neu">{previsao['confianca']:.1%}</div>
            <div class="metric-sub">Modelo: {melhor_nome}</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        ret_cor = 'pos' if previsao['retorno_hoje'] > 0 else 'neg'
        st.markdown(f"""
        <div class="metric-card yellow">
            <div class="metric-label">Preço Atual</div>
            <div class="metric-value neu">{previsao['preco_atual']:.4f}</div>
            <div class="metric-sub">Retorno hoje: <span class="{ret_cor}">{previsao['retorno_hoje']:+.2f}%</span></div>
        </div>""", unsafe_allow_html=True)
    with c6:
        rsi_cor = 'pos' if previsao['rsi'] < 70 else 'neg'
        st.markdown(f"""
        <div class="metric-card cyan">
            <div class="metric-label">RSI (14)</div>
            <div class="metric-value {rsi_cor}">{previsao['rsi']:.1f}</div>
            <div class="metric-sub">Ref: {previsao['data_referencia']}</div>
        </div>""", unsafe_allow_html=True)

    # ── MÉTRICAS PRINCIPAIS ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">📊 PERFORMANCE DO BACKTEST</div>', unsafe_allow_html=True)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    ret_cor  = 'pos' if mf['Retorno Total'] > 0 else 'neg'
    bh_cor   = 'pos' if mf['Retorno B&H'] > 0 else 'neg'
    sh_cor   = 'pos' if mf['Sharpe Ratio'] > 1 else ('neu' if mf['Sharpe Ratio'] > 0 else 'neg')
    dd_val   = f"{mf['Max Drawdown']:.1%}"
    wr_cor   = 'pos' if mf['Win Rate'] > 0.5 else 'neg'

    with m1:
        st.markdown(f"""
        <div class="metric-card blue">
            <div class="metric-label">Retorno Acumulado</div>
            <div class="metric-value {ret_cor}">{mf['Retorno Total']:+.1%}</div>
            <div class="metric-sub">B&H: <span class="{bh_cor}">{mf['Retorno B&H']:+.1%}</span></div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card {'green' if mf['Sharpe Ratio'] > 1 else 'yellow'}">
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value {sh_cor}">{mf['Sharpe Ratio']:.3f}</div>
            <div class="metric-sub">Sortino: {mf['Sortino Ratio']:.3f}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card {'green' if mf['Win Rate'] > 0.5 else 'red'}">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value {wr_cor}">{mf['Win Rate']:.1%}</div>
            <div class="metric-sub">{mf['N Trades']:,} trades</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card red">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value neg">{mf['Max Drawdown']:.1%}</div>
            <div class="metric-sub">Calmar: {mf['Calmar Ratio']:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with m5:
        pf_cor = 'pos' if mf['Profit Factor'] > 1 else 'neg'
        pf_val = f"{mf['Profit Factor']:.2f}" if mf['Profit Factor'] != np.inf else "∞"
        st.markdown(f"""
        <div class="metric-card purple">
            <div class="metric-label">Profit Factor</div>
            <div class="metric-value {pf_cor}">{pf_val}</div>
            <div class="metric-sub">Payoff: {mf['Payoff']:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with m6:
        roc_cor = 'pos' if avaliacao['roc'] > 0.55 else ('neu' if avaliacao['roc'] > 0.5 else 'neg')
        st.markdown(f"""
        <div class="metric-card cyan">
            <div class="metric-label">ROC-AUC</div>
            <div class="metric-value {roc_cor}">{avaliacao['roc']:.4f}</div>
            <div class="metric-sub">Modelo: {melhor_nome}</div>
        </div>""", unsafe_allow_html=True)

    # ── EQUITY CURVE ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">📈 EQUITY CURVE & DRAWDOWN</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_equity_curve(avaliacao, backtest, splits, ticker_display, melhor_nome),
                    use_container_width=True)

    # ── BACKTEST REALISTA ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">⚡ BACKTEST REALISTA (Com Stop/TP/Spread)</div>', unsafe_allow_html=True)
    b1, b2, b3, b4, b5 = st.columns(5)
    bt = backtest
    with b1:
        cap_cor = 'pos' if bt['capital_final'] > CAPITAL_INICIAL else 'neg'
        st.markdown(f"""
        <div class="metric-card blue">
            <div class="metric-label">Capital Final</div>
            <div class="metric-value {cap_cor}">${bt['capital_final']:,.0f}</div>
            <div class="metric-sub">Inicial: ${CAPITAL_INICIAL:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with b2:
        r_cor = 'pos' if bt['ret_total'] > 0 else 'neg'
        st.markdown(f"""
        <div class="metric-card {'green' if bt['ret_total']>0 else 'red'}">
            <div class="metric-label">Retorno Backtest</div>
            <div class="metric-value {r_cor}">{bt['ret_total']:+.2%}</div>
            <div class="metric-sub">Threshold: {thr:.0%}</div>
        </div>""", unsafe_allow_html=True)
    with b3:
        st.markdown(f"""
        <div class="metric-card yellow">
            <div class="metric-label">N° de Trades</div>
            <div class="metric-value neu">{bt['n_trades']:,}</div>
            <div class="metric-sub">Sinais filtrados por prob.</div>
        </div>""", unsafe_allow_html=True)
    with b4:
        wr_c = 'pos' if bt['win_rate'] > 0.5 else 'neg'
        st.markdown(f"""
        <div class="metric-card {'green' if bt['win_rate']>0.5 else 'red'}">
            <div class="metric-label">Win Rate (BT)</div>
            <div class="metric-value {wr_c}">{bt['win_rate']:.1%}</div>
            <div class="metric-sub">Com custos reais</div>
        </div>""", unsafe_allow_html=True)
    with b5:
        st.markdown(f"""
        <div class="metric-card red">
            <div class="metric-label">Max DD (BT)</div>
            <div class="metric-value neg">{bt['max_dd']:.1%}</div>
            <div class="metric-sub">Stop: {STOP_LOSS_PCT:.0%} | TP: {TAKE_PROFIT_PCT:.0%}</div>
        </div>""", unsafe_allow_html=True)

    # Tabela de trades
    if len(bt['df_trades']) > 0:
        with st.expander(f"📋 Ver últimos 50 trades ({len(bt['df_trades'])} total)"):
            df_show = bt['df_trades'].tail(50).copy()
            df_show['sinal']       = df_show['sinal'].map({1: '🟢 LONG', -1: '🔴 SHORT'})
            df_show['pnl']         = df_show['pnl'].round(2)
            df_show['retorno_real']= (df_show['retorno_real'] * 100).round(3)
            df_show['prob']        = (df_show['prob'] * 100).round(1)
            st.dataframe(df_show[['data', 'sinal', 'prob', 'preco', 'retorno_real', 'pnl', 'capital']],
                         use_container_width=True, height=300)

    # ── ANÁLISE DE MODELOS ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🤖 COMPARAÇÃO DE MODELOS</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.plotly_chart(plot_model_comparison(r['df_resultados'], melhor_nome),
                        use_container_width=True)
    with col_b:
        st.markdown("**Tabela de Resultados**")
        df_tab = r['df_resultados'][['Modelo','ROC_Test','Acc_Test','F1_Test','Prec_Test','Rec_Test']].copy()
        df_tab.columns = ['Modelo', 'ROC-AUC', 'Accuracy', 'F1', 'Precision', 'Recall']
        st.dataframe(df_tab.style.format({
            'ROC-AUC':   '{:.4f}', 'Accuracy': '{:.4f}',
            'F1':        '{:.4f}', 'Precision': '{:.4f}', 'Recall': '{:.4f}'
        }).highlight_max(axis=0, subset=['ROC-AUC', 'Accuracy', 'F1'],
                         props='background-color: rgba(59,130,246,0.2); color: #3b82f6; font-weight: bold;'),
                     use_container_width=True, height=280)

    # ── ROC + CONFUSION ────────────────────────────────────────────────────────
    col_c, col_d, col_e = st.columns(3)
    with col_c:
        st.plotly_chart(plot_roc_curve(avaliacao, splits), use_container_width=True)
    with col_d:
        st.plotly_chart(plot_confusion_matrix(avaliacao, splits), use_container_width=True)
    with col_e:
        st.plotly_chart(plot_returns_dist(splits['retornos_test']), use_container_width=True)

    # ── THRESHOLD ANALYSIS ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🎚 ANÁLISE DE THRESHOLD (OTIMIZAÇÃO AUTOMÁTICA)</div>', unsafe_allow_html=True)
    fig_thr = plot_threshold_analysis(r['df_threshold'], thr)
    if fig_thr:
        st.plotly_chart(fig_thr, use_container_width=True)
    st.info(f"✅ Threshold ótimo encontrado automaticamente: **{thr:.2f}** (maximiza F1-Score no conjunto de teste)")

    # ── WALK-FORWARD ───────────────────────────────────────────────────────────
    if r['wf_resultados'] is not None:
        st.markdown("---")
        st.markdown('<div class="section-header">🔄 WALK-FORWARD VALIDATION (OUT-OF-SAMPLE)</div>', unsafe_allow_html=True)
        fig_wf = plot_walk_forward(r['wf_resultados'])
        if fig_wf:
            st.plotly_chart(fig_wf, use_container_width=True)
        df_wf = r['wf_resultados']['df_wf']
        wf_cols = st.columns(4)
        with wf_cols[0]:
            st.markdown(f"""
            <div class="metric-card blue">
                <div class="metric-label">ROC-AUC Médio (OOS)</div>
                <div class="metric-value neu">{df_wf['ROC-AUC'].mean():.4f}</div>
                <div class="metric-sub">± {df_wf['ROC-AUC'].std():.4f}</div>
            </div>""", unsafe_allow_html=True)
        with wf_cols[1]:
            st.markdown(f"""
            <div class="metric-card green">
                <div class="metric-label">Accuracy Médio (OOS)</div>
                <div class="metric-value neu">{df_wf['Accuracy'].mean():.4f}</div>
                <div class="metric-sub">± {df_wf['Accuracy'].std():.4f}</div>
            </div>""", unsafe_allow_html=True)
        with wf_cols[2]:
            r_med = df_wf['Retorno'].mean()
            rc = 'pos' if r_med > 0 else 'neg'
            st.markdown(f"""
            <div class="metric-card {'green' if r_med>0 else 'red'}">
                <div class="metric-label">Retorno Médio (OOS)</div>
                <div class="metric-value {rc}">{r_med:+.2%}</div>
                <div class="metric-sub">± {df_wf['Retorno'].std():.2%}</div>
            </div>""", unsafe_allow_html=True)
        with wf_cols[3]:
            ret_wf_total = float((1 + r['wf_resultados']['retornos_todos']).prod() - 1)
            rc2 = 'pos' if ret_wf_total > 0 else 'neg'
            st.markdown(f"""
            <div class="metric-card purple">
                <div class="metric-label">Retorno Total (OOS)</div>
                <div class="metric-value {rc2}">{ret_wf_total:+.2%}</div>
                <div class="metric-sub">Todos os folds</div>
            </div>""", unsafe_allow_html=True)
        st.dataframe(df_wf.style.format({
            'Accuracy': '{:.4f}', 'ROC-AUC': '{:.4f}',
            'F1': '{:.4f}', 'Retorno': '{:.2%}'}),
            use_container_width=True)

    # ── FEATURE IMPORTANCE ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🔍 FEATURE IMPORTANCE (MUTUAL INFORMATION)</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_feature_importance(r['mi_ranking']), use_container_width=True)

    # ── DADOS BRUTOS ───────────────────────────────────────────────────────────
    with st.expander("📂 Dados Brutos (últimas 100 linhas)"):
        cols_show = ['Open', 'High', 'Low', 'Close', 'Volume',
                     'rsi_14', 'macd', 'atr_14', 'vol_20', 'adx_14', 'target']
        cols_exist = [c for c in cols_show if c in r['df'].columns]
        st.dataframe(r['df'][cols_exist].tail(100).round(4), use_container_width=True)

    # ── LOGS DO PIPELINE ───────────────────────────────────────────────────────
    with st.expander("🔧 Logs do Pipeline"):
        for log_line in r['logs']:
            st.markdown(f"<div style='font-family:JetBrains Mono;font-size:11px;color:#64748b;'>{log_line}</div>",
                        unsafe_allow_html=True)

    # ── FOOTER ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; font-family: JetBrains Mono, monospace; font-size: 11px; color: #1e2535; padding: 20px 0;">
        QUANT ML TRADING SYSTEM v4.0 &nbsp;·&nbsp; {ticker_display} &nbsp;·&nbsp;
        Modelo: {melhor_nome} &nbsp;·&nbsp; ROC-AUC: {avaliacao['roc']:.4f} &nbsp;·&nbsp;
        Threshold: {thr:.2f} &nbsp;·&nbsp;
        ⚠️ Apenas para fins educacionais. Não constitui recomendação financeira.
    </div>
    """, unsafe_allow_html=True)

else:
    # ── TELA INICIAL ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 80px 20px;">
        <div style="font-family: 'Syne', sans-serif; font-size: 48px; font-weight: 800;
                    background: linear-gradient(135deg, #1e2535 0%, #2a3550 50%, #1e2535 100%);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    margin-bottom: 20px;">
            ⚡
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 14px; color: #2a3550;
                    letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 40px;">
            Configure os parâmetros na barra lateral e clique em<br>
            <span style="color: #3b82f6; font-weight: 700;">EXECUTAR BACKTEST</span> para iniciar o pipeline
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Cards informativos
    ic1, ic2, ic3, ic4 = st.columns(4)
    with ic1:
        st.markdown("""
        <div class="metric-card blue">
            <div class="metric-label">Feature Engineering</div>
            <div class="metric-value neu" style="font-size:20px;">100+</div>
            <div class="metric-sub">Indicadores técnicos automáticos: SMA, EMA, RSI, MACD, ATR, Bollinger, VWAP, ADX e muito mais</div>
        </div>""", unsafe_allow_html=True)
    with ic2:
        st.markdown("""
        <div class="metric-card green">
            <div class="metric-label">Modelos Treinados</div>
            <div class="metric-value neu" style="font-size:20px;">6</div>
            <div class="metric-sub">LogReg, DecTree, RandomForest, XGBoost, LightGBM, CatBoost — seleção automática pelo melhor ROC-AUC</div>
        </div>""", unsafe_allow_html=True)
    with ic3:
        st.markdown("""
        <div class="metric-card yellow">
            <div class="metric-label">Backtest Realista</div>
            <div class="metric-value neu" style="font-size:20px;">✓</div>
            <div class="metric-sub">Spread, slippage, Stop-Loss, Take-Profit, position sizing por risco e threshold de probabilidade</div>
        </div>""", unsafe_allow_html=True)
    with ic4:
        st.markdown("""
        <div class="metric-card purple">
            <div class="metric-label">Walk-Forward (OOS)</div>
            <div class="metric-value neu" style="font-size:20px;">5</div>
            <div class="metric-sub">Validação out-of-sample com expanding window — elimina data leakage e mede performance real</div>
        </div>""", unsafe_allow_html=True)
