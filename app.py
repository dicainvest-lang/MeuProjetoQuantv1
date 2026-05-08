# -*- coding: utf-8 -*-
# ==============================================================================
# SISTEMA QUANTITATIVO DE ML PARA TRADING — DASHBOARD STREAMLIT
# Versão: 4.0 Dashboard | Todos os parâmetros de backtest na sidebar
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
import time

# ── ML ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================

st.set_page_config(
    page_title="Quant ML Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# ESTILOS DARK MODE
# ==============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');

/* Fundo principal */
.stApp {
    background-color: #0b0e14;
    font-family: 'Sora', sans-serif;
    color: #c9d1d9;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #58a6ff;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border-bottom: 1px solid #21262d;
    padding-bottom: 6px;
    margin-top: 18px;
}

/* Cards de métricas */
.metric-card {
    background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #58a6ff44; }
.metric-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #f0f6fc;
    line-height: 1;
}
.metric-delta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    margin-top: 5px;
}
.delta-pos { color: #3fb950; }
.delta-neg { color: #f85149; }
.delta-neu { color: #8b949e; }

/* Título do dashboard */
.dash-title {
    font-family: 'Sora', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #f0f6fc;
    letter-spacing: -0.02em;
    margin-bottom: 2px;
}
.dash-subtitle {
    font-size: 0.82rem;
    color: #8b949e;
    margin-bottom: 20px;
}

/* Seção */
.section-header {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #58a6ff;
    border-bottom: 1px solid #21262d;
    padding-bottom: 8px;
    margin: 24px 0 14px 0;
}

/* Sinal de previsão */
.signal-buy {
    background: linear-gradient(135deg, #1a2e1a 0%, #0d1117 100%);
    border: 2px solid #3fb950;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.signal-sell {
    background: linear-gradient(135deg, #2e1a1a 0%, #0d1117 100%);
    border: 2px solid #f85149;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.signal-neutral {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
    border: 2px solid #8b949e;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.signal-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
}
.signal-prob {
    font-size: 0.8rem;
    color: #8b949e;
    margin-top: 6px;
}

/* Tabela de modelos */
.model-table { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }

/* Info boxes */
.info-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 3px solid #58a6ff;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 0.8rem;
    color: #8b949e;
    margin: 8px 0;
}
.warning-box {
    background: #1c1a12;
    border: 1px solid #2d2700;
    border-left: 3px solid #d29922;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 0.8rem;
    color: #8b949e;
    margin: 8px 0;
}

/* Botão principal */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Sora', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 10px 24px;
    width: 100%;
    transition: opacity 0.2s;
}
div[data-testid="stButton"] > button:hover { opacity: 0.85; }

/* Sliders e inputs */
[data-testid="stSlider"] label,
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label,
[data-testid="stMultiSelect"] label {
    color: #8b949e !important;
    font-size: 0.78rem !important;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# FUNÇÕES DE DADOS E ML
# ==============================================================================

@st.cache_data(ttl=60)
def baixar_dados(ticker: str, anos: int = 5, interval: str = "1d") -> pd.DataFrame:
    data_inicio = (datetime.now() - timedelta(days=anos * 365)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=data_inicio, end=datetime.now().strftime("%Y-%m-%d"),
                     interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().capitalize() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    if 'Volume' not in df.columns:
        df['Volume'] = 0
    df['Volume'] = df['Volume'].fillna(0).astype(float)
    df.dropna(subset=['Open','High','Low','Close'], inplace=True)
    for col in ['Open','High','Low','Close']:
        df[col] = df[col].astype(float)
    df = df[df['Close'] > 0]
    return df.sort_index()


def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering completo com indicadores técnicos."""
    df = df.copy()
    o, h, l, c = df['Open'], df['High'], df['Low'], df['Close']
    v = df['Volume'].replace(0, np.nan)

    # Target
    df['retorno_futuro_log'] = np.log(c.shift(-1) / c)
    df['retorno_futuro_pct'] = c.pct_change().shift(-1) * 100
    df['target'] = (df['retorno_futuro_log'] > 0).astype(int)
    df['retorno_atual'] = np.log(c / c.shift(1))

    # Médias móveis
    for p in [5, 10, 20, 50, 100, 200]:
        df[f'sma_{p}'] = c.rolling(p).mean()
        df[f'ema_{p}'] = c.ewm(span=p, adjust=False).mean()

    for p in [5, 10, 20, 50, 200]:
        df[f'preco_sma_{p}_ratio'] = c / df[f'sma_{p}'] - 1

    df['cruz_sma_5_20']   = (df['sma_5']  > df['sma_20']).astype(int)
    df['cruz_sma_20_50']  = (df['sma_20'] > df['sma_50']).astype(int)
    df['cruz_sma_50_200'] = (df['sma_50'] > df['sma_200']).astype(int)

    # VWAP
    tp = (h + l + c) / 3
    df['vwap'] = (tp * v).rolling(20).sum() / v.rolling(20).sum()
    df['preco_vwap_ratio'] = c / df['vwap'] - 1

    # RSI
    for p in [7, 14, 21]:
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(p).mean()
        loss  = (-delta.clip(upper=0)).rolling(p).mean()
        rs    = gain / loss.replace(0, np.nan)
        df[f'rsi_{p}'] = 100 - (100 / (1 + rs))
        df[f'rsi_{p}_norm'] = df[f'rsi_{p}'] / 50 - 1

    # ROC
    for p in [5, 10, 20, 60]:
        df[f'roc_{p}'] = c.pct_change(p) * 100

    # Momentum
    for p in [5, 10, 20]:
        df[f'mom_{p}'] = c - c.shift(p)
        df[f'mom_{p}_norm'] = df[f'mom_{p}'] / c.shift(p)

    # Stochastic
    for p in [14, 21]:
        low_min  = l.rolling(p).min()
        high_max = h.rolling(p).max()
        denom    = high_max - low_min
        df[f'stoch_k_{p}'] = 100 * (c - low_min) / denom.replace(0, np.nan)
        df[f'stoch_d_{p}'] = df[f'stoch_k_{p}'].rolling(3).mean()
        df[f'stoch_diff_{p}'] = df[f'stoch_k_{p}'] - df[f'stoch_d_{p}']

    # ATR
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    for p in [7, 14, 21]:
        df[f'atr_{p}']      = true_range.ewm(span=p, adjust=False).mean()
        df[f'atr_{p}_norm'] = df[f'atr_{p}'] / c

    # Volatilidade
    ret = c.pct_change()
    for p in [5, 10, 20, 60]:
        df[f'vol_{p}'] = ret.rolling(p).std() * np.sqrt(252)

    # Bollinger
    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    df['bb_upper_20'] = mid + 2 * std
    df['bb_lower_20'] = mid - 2 * std
    df['bb_width_20'] = (df['bb_upper_20'] - df['bb_lower_20']) / mid
    df['bb_pos_20']   = (c - df['bb_lower_20']) / (df['bb_upper_20'] - df['bb_lower_20'] + 1e-9)

    df['vol_regime'] = df['vol_5'] / df['vol_20']

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df['macd']        = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']
    df['macd_norm']   = df['macd'] / c

    # ADX
    dm_plus  = h.diff().clip(lower=0)
    dm_minus = (-l.diff()).clip(lower=0)
    tr_s     = true_range.ewm(span=14, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm(span=14, adjust=False).mean() / tr_s.replace(0, np.nan)
    di_minus = 100 * dm_minus.ewm(span=14, adjust=False).mean() / tr_s.replace(0, np.nan)
    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9)
    df['adx_14']      = dx.ewm(span=14, adjust=False).mean()
    df['di_plus_14']  = di_plus
    df['di_minus_14'] = di_minus
    df['di_diff_14']  = di_plus - di_minus

    for p in [20, 50]:
        df[f'ema_{p}_slope'] = df[f'ema_{p}'].diff(5) / df[f'ema_{p}'].shift(5)

    # Z-Score
    for p in [20, 60]:
        mu = c.rolling(p).mean()
        sg = c.rolling(p).std()
        df[f'zscore_{p}'] = (c - mu) / sg.replace(0, np.nan)

    for p in [20, 50, 200]:
        df[f'dist_ema_{p}'] = (c / df[f'ema_{p}'] - 1) * 100

    # Volume
    df['vol_rel_20'] = v / v.rolling(20).mean()
    df['vol_rel_50'] = v / v.rolling(50).mean()
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    df['obv']       = obv
    df['obv_sma20'] = obv.rolling(20).mean()
    df['obv_ratio'] = obv / (df['obv_sma20'] + 1e-9)
    mf_vol = ((c - l) - (h - c)) / (h - l + 1e-9) * v
    df['mf_20'] = mf_vol.rolling(20).sum() / v.rolling(20).sum()

    # Candlestick
    df['corpo']       = (c - o).abs()
    df['corpo_pct']   = df['corpo'] / o.clip(lower=1e-9)
    df['sombra_sup']  = h - pd.concat([c, o], axis=1).max(axis=1)
    df['sombra_inf']  = pd.concat([c, o], axis=1).min(axis=1) - l
    df['sombra_sup_pct'] = df['sombra_sup'] / o.clip(lower=1e-9)
    df['sombra_inf_pct'] = df['sombra_inf'] / o.clip(lower=1e-9)
    df['range']       = h - l
    df['range_pct']   = df['range'] / o.clip(lower=1e-9)
    df['gap']         = (o - c.shift(1)) / c.shift(1).clip(lower=1e-9)
    df['direcao_candle'] = np.sign(c - o)

    for n in [3, 5]:
        direcoes = (c > o).astype(int)
        df[f'sequencia_alta_{n}'] = direcoes.rolling(n).sum()
        df[f'pct_alta_{n}']       = df[f'sequencia_alta_{n}'] / n

    # Lags
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f'ret_lag_{lag}'] = ret.shift(lag)
    for lag in [1, 5, 10]:
        df[f'vol20_lag_{lag}']  = df['vol_20'].shift(lag)
        df[f'rsi14_lag_{lag}']  = df['rsi_14'].shift(lag)
    for lag in [1, 3]:
        df[f'macd_lag_{lag}'] = df['macd'].shift(lag)

    # Rolling stats
    for p in [20, 60]:
        df[f'skew_{p}']  = ret.rolling(p).skew()
        df[f'kurt_{p}']  = ret.rolling(p).kurt()
        df[f'maxdd_{p}'] = c.rolling(p).apply(
            lambda x: (x[-1] / x.max() - 1) if len(x) > 0 else 0, raw=True)
        df[f'autocorr_{p}'] = ret.rolling(p).apply(
            lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 1 else 0, raw=True)

    # Regime
    above_sma50  = (c > df['sma_50']).astype(int)
    above_sma200 = (c > df['sma_200']).astype(int)
    df['regime_tendencia'] = (df['adx_14'] > 25).astype(int)
    df['regime_alta_vol']  = (df['vol_regime'] > 1.2).astype(int)
    df['regime_bull'] = ((above_sma50 == 1) & (above_sma200 == 1)).astype(int)
    df['regime_bear'] = ((above_sma50 == 0) & (above_sma200 == 0)).astype(int)

    # Tempo
    df['dia_semana'] = df.index.dayofweek
    df['dia_mes']    = df.index.day
    df['mes']        = df.index.month
    df['trimestre']  = df.index.quarter
    df['fim_semana'] = (df.index.dayofweek >= 3).astype(int)
    df['inicio_mes'] = (df.index.day <= 5).astype(int)
    df['fim_mes']    = (df.index.day >= 25).astype(int)

    # Remover última linha (sem target válido)
    df = df.iloc[:-1]
    return df


COLUNAS_NAO_FEATURE = [
    'Open','High','Low','Close','Volume',
    'target','retorno_futuro_log','retorno_futuro_pct',
    'retorno_atual'
]


def selecionar_features(df: pd.DataFrame, max_features: int = 50) -> list:
    features_cand = [c for c in df.columns if c not in COLUNAS_NAO_FEATURE]
    df_f = df[features_cand + ['target']].copy()
    nan_pct = df_f.isnull().mean()
    features_ok = nan_pct[nan_pct < 0.30].index.tolist()
    features_ok = [f for f in features_ok if f != 'target']
    df_clean = df_f[features_ok + ['target']].dropna()
    var = df_clean[features_ok].var()
    features_ok = var[var > 1e-10].index.tolist()

    if len(features_ok) == 0:
        return []

    corr = df_clean[features_ok].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    rem = {col for col in upper.columns if any(upper[col] > 0.95)}
    features_ok = [f for f in features_ok if f not in rem]

    X = df_clean[features_ok].values
    y = df_clean['target'].values
    mi = mutual_info_classif(X, y, random_state=42)
    mi_rank = pd.Series(mi, index=features_ok).sort_values(ascending=False)
    return mi_rank.head(max_features).index.tolist()


def treinar_modelos(X_train, y_train, X_val, y_val, X_test, y_test):
    modelos = {}
    resultados = []

    modelos['LogReg'] = LogisticRegression(C=0.1, max_iter=1000, random_state=42,
                                            class_weight='balanced')
    modelos['RandForest'] = RandomForestClassifier(n_estimators=150, max_depth=6,
                                                    min_samples_leaf=10, max_features='sqrt',
                                                    random_state=42, class_weight='balanced',
                                                    n_jobs=-1)
    if XGB_AVAILABLE:
        modelos['XGBoost'] = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                            subsample=0.8, colsample_bytree=0.8, gamma=0.1,
                                            eval_metric='logloss', random_state=42,
                                            n_jobs=-1, verbosity=0)
    if LGB_AVAILABLE:
        modelos['LightGBM'] = LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                              subsample=0.8, colsample_bytree=0.8,
                                              min_child_samples=20, class_weight='balanced',
                                              random_state=42, n_jobs=-1, verbose=-1)

    modelos_treinados = {}
    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred_t = modelo.predict(X_test)
        y_prob_t = modelo.predict_proba(X_test)[:, 1]
        resultados.append({
            'Modelo': nome,
            'ROC_Test': roc_auc_score(y_test, y_prob_t),
            'Acc_Test': accuracy_score(y_test, y_pred_t),
            'F1_Test':  f1_score(y_test, y_pred_t, zero_division=0),
            'Prec_Test': precision_score(y_test, y_pred_t, zero_division=0),
            'Rec_Test':  recall_score(y_test, y_pred_t, zero_division=0),
        })
        modelos_treinados[nome] = {'modelo': modelo,
                                    'y_pred_test': y_pred_t,
                                    'y_prob_test': y_prob_t}

    df_res = pd.DataFrame(resultados).sort_values('ROC_Test', ascending=False)
    return df_res, modelos_treinados


def backtest_realista(y_pred, y_prob, closes, retornos, datas,
                      capital_inicial, prob_threshold, risco_trade,
                      stop_loss_pct, take_profit_pct,
                      spread_pips, slippage, custo_operacional):
    capital = capital_inicial
    equity_curve = [capital]
    trades = []

    for i in range(len(y_pred)):
        preco = closes[i]
        ret_real = retornos[i] / 100

        sinal = 0
        if y_prob[i] >= prob_threshold:
            sinal = 1
        elif y_prob[i] <= (1 - prob_threshold):
            sinal = -1

        if sinal == 0:
            equity_curve.append(capital)
            continue

        custo_ent = (spread_pips + slippage + custo_operacional) * capital
        tamanho = capital * risco_trade
        pnl = tamanho * sinal * ret_real

        if sinal == 1:
            if ret_real < -stop_loss_pct:
                pnl = -tamanho * stop_loss_pct
            elif ret_real > take_profit_pct:
                pnl = tamanho * take_profit_pct
        else:
            if ret_real > stop_loss_pct:
                pnl = -tamanho * stop_loss_pct
            elif ret_real < -take_profit_pct:
                pnl = tamanho * take_profit_pct

        pnl_liq = pnl - custo_ent
        capital += pnl_liq
        equity_curve.append(capital)
        trades.append({'data': datas[i], 'sinal': sinal, 'prob': y_prob[i],
                        'preco': preco, 'retorno_real': ret_real,
                        'pnl': pnl_liq, 'capital': capital})

    eq = pd.Series(equity_curve)
    roll_max = eq.cummax()
    dd = (eq - roll_max) / roll_max
    max_dd = dd.min()
    ret_total = capital / capital_inicial - 1

    df_trades = pd.DataFrame(trades)
    n_trades = len(df_trades)
    if n_trades > 0:
        wins = (df_trades['pnl'] > 0).sum()
        win_rate = wins / n_trades
        gp = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
        gl = abs(df_trades[df_trades['pnl'] < 0]['pnl'].sum())
        pf = gp / gl if gl > 0 else float('inf')
    else:
        win_rate = pf = 0

    # Sharpe
    eq_rets = eq.pct_change().dropna()
    sharpe = (eq_rets.mean() / eq_rets.std() * np.sqrt(252)) if eq_rets.std() > 0 else 0

    return {
        'equity_curve': eq,
        'df_trades': df_trades,
        'ret_total': ret_total,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'profit_factor': pf,
        'max_dd': max_dd,
        'capital_final': capital,
        'sharpe': sharpe,
        'drawdown': dd,
    }


def calcular_metricas_financeiras(retornos_mod, retornos_bh, capital_inicial):
    equity    = pd.Series(capital_inicial * (1 + retornos_mod).cumprod())
    equity_bh = pd.Series(capital_inicial * (1 + retornos_bh).cumprod())
    ret_total = equity.iloc[-1] / capital_inicial - 1
    ret_bh    = equity_bh.iloc[-1] / capital_inicial - 1
    std       = retornos_mod.std()
    sharpe    = (retornos_mod.mean() / std * np.sqrt(252)) if std > 0 else 0
    downside  = retornos_mod[retornos_mod < 0].std()
    sortino   = (retornos_mod.mean() / downside * np.sqrt(252)) if downside > 0 else 0
    roll_max  = equity.cummax()
    drawdown  = (equity - roll_max) / roll_max
    max_dd    = drawdown.min()
    rm = pd.Series(retornos_mod)
    wins   = (rm > 0).sum()
    losses = (rm <= 0).sum()
    win_rate = wins / len(rm) if len(rm) > 0 else 0
    gp = rm[rm > 0].sum()
    gl = abs(rm[rm < 0].sum())
    pf = gp / gl if gl > 0 else float('inf')
    avg_win  = rm[rm > 0].mean() if wins > 0 else 0
    avg_loss = abs(rm[rm < 0].mean()) if losses > 0 else 1
    payoff   = avg_win / avg_loss if avg_loss > 0 else 0
    return {
        'Retorno Total': ret_total, 'Retorno B&H': ret_bh,
        'Sharpe Ratio': sharpe, 'Sortino Ratio': sortino,
        'Max Drawdown': max_dd, 'Win Rate': win_rate,
        'Profit Factor': pf, 'Payoff': payoff,
        'equity': equity, 'equity_bh': equity_bh, 'drawdown': drawdown,
    }


# ==============================================================================
# SIDEBAR — TODOS OS PARÂMETROS EDITÁVEIS
# ==============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Configurações")

    # ── ATIVO & DADOS ────────────────────────────────────────────────────────
    st.markdown("### 📌 Ativo & Dados")

    TICKERS_SUGERIDOS = [
        "USDJPY=X", "EURUSD=X", "GBPUSD=X", "BTCUSDT-USD", "BTC-USD",
        "^GSPC", "^IXIC", "^BVSP", "GC=F", "CL=F",
        "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META",
        "PETR4.SA", "VALE3.SA", "ITUB4.SA",
    ]
    TICKER = st.selectbox("Ativo (Ticker)", options=TICKERS_SUGERIDOS,
                          index=0, help="Selecione ou insira o símbolo")
    TICKER_CUSTOM = st.text_input("Ou insira um ticker personalizado:", value="",
                                  placeholder="Ex: AAPL, EURUSD=X, BTC-USD")
    if TICKER_CUSTOM.strip():
        TICKER = TICKER_CUSTOM.strip().upper()

    ANOS_HISTORICO = st.slider("Anos de histórico", 1, 15, 5,
                                help="Quantidade de anos de dados para download")

    TIMEFRAME_MAP = {"Diário (1d)": "1d", "Semanal (1wk)": "1wk", "Mensal (1mo)": "1mo"}
    TIMEFRAME = TIMEFRAME_MAP[st.selectbox("Timeframe", list(TIMEFRAME_MAP.keys()))]

    MAX_FEATURES = st.slider("Nº máx. de features", 20, 80, 50,
                             help="Quantidade de features selecionadas por Mutual Information")

    # ── SPLIT TEMPORAL ──────────────────────────────────────────────────────
    st.markdown("### 📊 Split Temporal")
    col1s, col2s = st.columns(2)
    with col1s:
        TRAIN_RATIO = st.number_input("Treino %", 0.50, 0.85, 0.70, 0.05,
                                      format="%.2f")
    with col2s:
        VAL_RATIO = st.number_input("Val %", 0.05, 0.25, 0.15, 0.05,
                                    format="%.2f")
    TEST_RATIO = round(1.0 - TRAIN_RATIO - VAL_RATIO, 2)
    st.caption(f"🔢 Teste: **{TEST_RATIO:.0%}** (calculado automaticamente)")
    if TEST_RATIO <= 0:
        st.error("⚠ Treino + Val não pode ultrapassar 100%!")

    # ── THRESHOLD DE PROBABILIDADE ──────────────────────────────────────────
    st.markdown("### 🎯 Threshold de Probabilidade")
    PROB_THRESHOLD = st.slider(
        "Prob. mínima para operar",
        min_value=0.50, max_value=0.90, value=0.55, step=0.01,
        format="%.2f",
        help="O modelo só entra em posição se a probabilidade for maior que este valor. "
             "Valores mais altos = menos trades, mais seletivo."
    )
    st.caption(f"Neutro se prob ∈ [{1-PROB_THRESHOLD:.2f}, {PROB_THRESHOLD:.2f}]")

    # ── GESTÃO DE RISCO ─────────────────────────────────────────────────────
    st.markdown("### 🛡️ Gestão de Risco")
    CAPITAL_INICIAL = st.number_input(
        "Capital inicial (USD)", 1_000, 10_000_000, 100_000, 1_000,
        help="Capital inicial para simulação do backtest"
    )
    RISCO_POR_TRADE = st.slider(
        "Risco por trade (%)", 0.005, 0.10, 0.02, 0.005, format="%.3f",
        help="Percentual do capital arriscado em cada operação (Kelly / position sizing)"
    )

    # ── STOP LOSS & TAKE PROFIT ─────────────────────────────────────────────
    st.markdown("### 🔴🟢 Stop Loss & Take Profit")
    STOP_LOSS_PCT = st.slider(
        "Stop Loss (%)", 0.005, 0.10, 0.02, 0.005, format="%.3f",
        help="Se o trade atingir essa perda, a posição é encerrada automaticamente"
    )
    TAKE_PROFIT_PCT = st.slider(
        "Take Profit (%)", 0.01, 0.20, 0.04, 0.005, format="%.3f",
        help="Se o trade atingir esse ganho, a posição é encerrada com lucro"
    )
    RR_RATIO = TAKE_PROFIT_PCT / STOP_LOSS_PCT if STOP_LOSS_PCT > 0 else 0
    st.caption(f"📐 Risco/Retorno: **1:{RR_RATIO:.1f}** "
               f"({'✅ Favorável' if RR_RATIO >= 2 else '⚠ Baixo'})")

    # ── CUSTOS OPERACIONAIS ─────────────────────────────────────────────────
    st.markdown("### 💸 Custos Operacionais")
    SPREAD_PIPS = st.number_input(
        "Spread (%)", 0.0, 0.005, 0.0002, 0.0001, format="%.4f",
        help="Spread do broker em percentual (ex: 0.0002 = 2 pips aprox.)"
    )
    SLIPPAGE = st.number_input(
        "Slippage (%)", 0.0, 0.005, 0.0001, 0.0001, format="%.4f",
        help="Derrapagem estimada na execução da ordem"
    )
    CUSTO_OPERACIONAL = st.number_input(
        "Custo operacional (%)", 0.0, 0.005, 0.0001, 0.0001, format="%.4f",
        help="Custo por operação (corretagem, impostos estimados etc.)"
    )
    custo_total_pct = (SPREAD_PIPS + SLIPPAGE + CUSTO_OPERACIONAL) * 100
    st.caption(f"💰 Custo total por operação: **{custo_total_pct:.4f}%**")

    # ── INDICADORES TÉCNICOS (PARÂMETROS) ────────────────────────────────────
    st.markdown("### 📈 Parâmetros dos Indicadores")
    with st.expander("🔧 Z-Score & Bollinger"):
        ZSCORE_JANELA_CURTA = st.slider("Z-Score — janela curta (dias)", 5, 40, 20,
            help="Janela do rolling Z-Score de curto prazo")
        ZSCORE_JANELA_LONGA = st.slider("Z-Score — janela longa (dias)", 30, 120, 60,
            help="Janela do rolling Z-Score de longo prazo")
        BOLLINGER_JANELA    = st.slider("Bollinger — janela (dias)", 10, 50, 20,
            help="Período da média central das Bandas de Bollinger")
        BOLLINGER_STD       = st.slider("Bollinger — desvios padrão", 1.0, 3.5, 2.0, 0.1,
            help="Multiplicador do desvio padrão para as bandas superior/inferior")
        REGIME_VOL_THRESH   = st.slider("Regime alta volatilidade (mult.)", 1.0, 2.5, 1.2, 0.1,
            help="Multiplier: vol_5 / vol_20 acima deste valor → regime de alta volatilidade")
        ADX_TENDENCIA_THRESH= st.slider("ADX — limiar de tendência", 15, 40, 25,
            help="ADX acima deste valor indica mercado em tendência forte")

    with st.expander("📉 RSI & Stochastic"):
        RSI_SOBRECOMPRADO   = st.slider("RSI — sobrecomprado", 60, 90, 70,
            help="Nível de RSI considerado sobrecomprado (venda potencial)")
        RSI_SOBREVENDIDO    = st.slider("RSI — sobrevendido",  10, 45, 30,
            help="Nível de RSI considerado sobrevendido (compra potencial)")

    with st.expander("🔄 Walk-Forward"):
        WF_N_SPLITS  = st.slider("Walk-Forward — nº de splits", 3, 10, 5,
            help="Número de janelas na validação walk-forward")
        WF_TEST_SIZE = st.slider("Walk-Forward — dias de teste/janela", 20, 120, 60,
            help="Quantidade de dias no período de teste de cada janela")

    with st.expander("🧠 Seleção de Features"):
        CORRELACAO_MAX = st.slider(
            "Correlação máx. entre features", 0.70, 0.99, 0.95, 0.01,
            help="Features com correlação de Pearson acima deste valor são removidas "
                 "(reduz multicolinearidade)"
        )
        NAN_THRESHOLD = st.slider(
            "Threshold NaN (%)", 0.10, 0.50, 0.30, 0.05,
            help="Features com mais do que este % de valores NaN são descartadas"
        )

    st.markdown("---")
    executar = st.button("🚀 Executar Backtest", use_container_width=True)
    st.markdown("""
    <div class='info-box'>
    ⏱ Dados atualizados automaticamente a cada 60s via cache.<br>
    Clique em <strong>Executar Backtest</strong> para re-treinar o modelo com os parâmetros atuais.
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# CABEÇALHO PRINCIPAL
# ==============================================================================

st.markdown(f"""
<div class='dash-title'>📊 Quant ML Trading Dashboard</div>
<div class='dash-subtitle'>
Sistema quantitativo de Machine Learning para previsão direcional · Ativo: <strong>{TICKER}</strong> · 
Dados: Yahoo Finance · Atualização a cada 60s
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# EXECUÇÃO DO PIPELINE
# ==============================================================================

if not executar and 'resultado' not in st.session_state:
    st.markdown("""
    <div class='warning-box'>
    ⚙️ Configure os parâmetros na barra lateral e clique em <strong>🚀 Executar Backtest</strong> para iniciar o pipeline de ML.
    </div>
    """, unsafe_allow_html=True)

    # Preview dos parâmetros configurados
    st.markdown("<div class='section-header'>📋 Parâmetros Configurados</div>", unsafe_allow_html=True)

    cols_prev = st.columns(4)
    params_preview = {
        "Ativo": TICKER,
        "Histórico": f"{ANOS_HISTORICO} anos",
        "Timeframe": TIMEFRAME,
        "Capital": f"${CAPITAL_INICIAL:,.0f}",
        "Stop Loss": f"{STOP_LOSS_PCT:.1%}",
        "Take Profit": f"{TAKE_PROFIT_PCT:.1%}",
        "Threshold": f"{PROB_THRESHOLD:.0%}",
        "Risco/Trade": f"{RISCO_POR_TRADE:.1%}",
        "Treino/Val/Teste": f"{TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%}",
        "Spread": f"{SPREAD_PIPS:.4f}",
        "Slippage": f"{SLIPPAGE:.4f}",
        "Max Features": MAX_FEATURES,
    }
    keys = list(params_preview.keys())
    for i, col in enumerate(cols_prev):
        for j in range(3):
            idx = i * 3 + j
            if idx < len(keys):
                col.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>{keys[idx]}</div>
                    <div class='metric-value' style='font-size:1.1rem;'>{params_preview[keys[idx]]}</div>
                </div>
                """, unsafe_allow_html=True)
    st.stop()

# ── Pipeline ──────────────────────────────────────────────────────────────────
if executar or 'resultado' not in st.session_state:

    progress_container = st.container()
    with progress_container:
        bar = st.progress(0, text="🔄 Iniciando pipeline...")

        try:
            # Passo 1: Download
            bar.progress(5, text=f"📥 Baixando dados de {TICKER}...")
            df_raw = baixar_dados(TICKER, ANOS_HISTORICO, TIMEFRAME)
            if df_raw is None or len(df_raw) < 200:
                st.error(f"❌ Dados insuficientes para {TICKER}. Verifique o ticker e tente novamente.")
                st.stop()

            # Passo 2: Features
            bar.progress(15, text="⚙️ Calculando indicadores técnicos e features...")
            df = construir_features(df_raw)
            df_raw_close = df_raw['Close'].copy()

            # Passo 3: Seleção de features
            bar.progress(30, text="🔍 Selecionando melhores features por Mutual Information...")
            features_sel = selecionar_features(df, max_features=MAX_FEATURES)
            if len(features_sel) == 0:
                st.error("❌ Nenhuma feature válida encontrada. Tente aumentar o período histórico.")
                st.stop()

            # Passo 4: Preparar dataset
            bar.progress(40, text="📐 Preparando dataset...")
            colunas_need = features_sel + ['target', 'Close', 'retorno_futuro_pct', 'retorno_atual']
            colunas_exist = [c for c in colunas_need if c in df.columns]
            df_modelo = df[colunas_exist].dropna()

            X = df_modelo[features_sel].values
            y = df_modelo['target'].values
            datas   = df_modelo.index
            closes  = df_modelo['Close'].values
            retornos = df_modelo['retorno_futuro_pct'].values if 'retorno_futuro_pct' in df_modelo.columns else np.zeros(len(df_modelo))

            # Passo 5: Split temporal
            n = len(X)
            n_train = int(n * TRAIN_RATIO)
            n_val   = int(n * VAL_RATIO)

            X_train, y_train = X[:n_train], y[:n_train]
            X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
            X_test,  y_test  = X[n_train+n_val:], y[n_train+n_val:]
            ret_test   = retornos[n_train+n_val:]
            closes_test = closes[n_train+n_val:]
            datas_test  = datas[n_train+n_val:]

            # Normalização
            scaler = RobustScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s   = scaler.transform(X_val)
            X_test_s  = scaler.transform(X_test)
            X_all_s   = scaler.transform(X)

            # Passo 6: Treinar modelos
            bar.progress(55, text="🧠 Treinando modelos de ML...")
            df_res, modelos_treinados = treinar_modelos(
                X_train_s, y_train, X_val_s, y_val, X_test_s, y_test)

            # Passo 7: Melhor modelo
            melhor_nome = df_res.iloc[0]['Modelo']
            melhor_modelo = modelos_treinados[melhor_nome]['modelo']
            y_pred_test = modelos_treinados[melhor_nome]['y_pred_test']
            y_prob_test = modelos_treinados[melhor_nome]['y_prob_test']

            # Passo 8: Backtest realista
            bar.progress(70, text="📈 Executando backtest realista...")
            bt = backtest_realista(
                y_pred_test, y_prob_test, closes_test, ret_test, datas_test,
                CAPITAL_INICIAL, PROB_THRESHOLD, RISCO_POR_TRADE,
                STOP_LOSS_PCT, TAKE_PROFIT_PCT,
                SPREAD_PIPS, SLIPPAGE, CUSTO_OPERACIONAL
            )

            # Passo 9: Métricas financeiras
            sinais = np.where(y_pred_test == 1, 1, -1)
            rets_mod = pd.Series(sinais * ret_test / 100)
            rets_bh  = pd.Series(ret_test / 100)
            metricas_fin = calcular_metricas_financeiras(rets_mod, rets_bh, CAPITAL_INICIAL)

            # Passo 10: Previsão próximo dia
            bar.progress(88, text="🔮 Gerando previsão para o próximo dia...")
            X_prod = X_all_s[-1:].copy()
            prob_alta  = melhor_modelo.predict_proba(X_prod)[0, 1]
            prob_queda = 1 - prob_alta
            if prob_alta >= PROB_THRESHOLD:
                sinal_hoje, sinal_code = "COMPRA 🟢", "BUY"
                confianca = prob_alta
            elif prob_queda >= PROB_THRESHOLD:
                sinal_hoje, sinal_code = "VENDA 🔴", "SELL"
                confianca = prob_queda
            else:
                sinal_hoje, sinal_code = "NEUTRO ⚪", "NEUTRAL"
                confianca = max(prob_alta, prob_queda)

            preco_atual   = closes[-1]
            rsi_atual     = df_modelo['rsi_14'].iloc[-1] if 'rsi_14' in df_modelo.columns else 0
            vol_20        = df_modelo['vol_20'].iloc[-1] if 'vol_20' in df_modelo.columns else 0
            data_ref      = datas[-1].date()
            ret_hoje      = df_modelo['retorno_atual'].iloc[-1] * 100 if 'retorno_atual' in df_modelo.columns else 0

            bar.progress(100, text="✅ Pipeline concluído!")
            time.sleep(0.4)
            bar.empty()

            # Salvar resultado na sessão
            st.session_state['resultado'] = {
                'df_res': df_res, 'melhor_nome': melhor_nome,
                'bt': bt, 'metricas_fin': metricas_fin,
                'y_pred_test': y_pred_test, 'y_prob_test': y_prob_test,
                'y_test': y_test, 'datas_test': datas_test,
                'ret_test': ret_test, 'closes_test': closes_test,
                'closes': closes, 'retornos': retornos, 'datas': datas,
                'df_modelo': df_modelo, 'features_sel': features_sel,
                'prob_alta': prob_alta, 'prob_queda': prob_queda,
                'sinal_hoje': sinal_hoje, 'sinal_code': sinal_code,
                'confianca': confianca, 'preco_atual': preco_atual,
                'rsi_atual': rsi_atual, 'vol_20': vol_20,
                'data_ref': data_ref, 'ret_hoje': ret_hoje,
                'n_features': len(features_sel),
                'n_amostras': n, 'periodo_inicio': datas[0].date(),
                'periodo_fim': datas[-1].date(),
                'ticker': TICKER,
            }

        except Exception as e:
            bar.empty()
            st.error(f"❌ Erro no pipeline: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()

# ==============================================================================
# EXIBIÇÃO DOS RESULTADOS
# ==============================================================================

if 'resultado' not in st.session_state:
    st.stop()

res = st.session_state['resultado']
bt  = res['bt']
mf  = res['metricas_fin']

# ── SINAL DE PREVISÃO ─────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>🔮 Sinal para o Próximo Dia</div>",
            unsafe_allow_html=True)

col_sig, col_info1, col_info2, col_info3, col_info4 = st.columns([2, 1, 1, 1, 1])

with col_sig:
    css_class = {'BUY': 'signal-buy', 'SELL': 'signal-sell', 'NEUTRAL': 'signal-neutral'}[res['sinal_code']]
    color = {'BUY': '#3fb950', 'SELL': '#f85149', 'NEUTRAL': '#8b949e'}[res['sinal_code']]
    st.markdown(f"""
    <div class='{css_class}'>
        <div class='signal-text' style='color:{color};'>{res['sinal_hoje']}</div>
        <div class='signal-prob'>Confiança: <strong>{res['confianca']:.1%}</strong> · Threshold: {PROB_THRESHOLD:.0%}</div>
        <div class='signal-prob'>Ref: {res['data_ref']} · {res['ticker']}</div>
    </div>
    """, unsafe_allow_html=True)

with col_info1:
    st.markdown(f"""<div class='metric-card'>
    <div class='metric-label'>Preço Atual</div>
    <div class='metric-value' style='font-size:1.2rem;'>{res['preco_atual']:.4f}</div>
    <div class='metric-delta {"delta-pos" if res["ret_hoje"] >= 0 else "delta-neg"}'>{res["ret_hoje"]:+.3f}% hoje</div>
    </div>""", unsafe_allow_html=True)

with col_info2:
    st.markdown(f"""<div class='metric-card'>
    <div class='metric-label'>Prob. Alta ↑</div>
    <div class='metric-value' style='font-size:1.2rem; color:#3fb950;'>{res['prob_alta']:.1%}</div>
    <div class='metric-delta delta-neu'>RSI(14): {res['rsi_atual']:.1f}</div>
    </div>""", unsafe_allow_html=True)

with col_info3:
    st.markdown(f"""<div class='metric-card'>
    <div class='metric-label'>Prob. Queda ↓</div>
    <div class='metric-value' style='font-size:1.2rem; color:#f85149;'>{res['prob_queda']:.1%}</div>
    <div class='metric-delta delta-neu'>Vol(20): {res['vol_20']:.4f}</div>
    </div>""", unsafe_allow_html=True)

with col_info4:
    st.markdown(f"""<div class='metric-card'>
    <div class='metric-label'>Melhor Modelo</div>
    <div class='metric-value' style='font-size:1.1rem;'>{res['melhor_nome']}</div>
    <div class='metric-delta delta-neu'>{res['n_features']} features</div>
    </div>""", unsafe_allow_html=True)

# ── MÉTRICAS DO BACKTEST ──────────────────────────────────────────────────────
st.markdown("<div class='section-header'>📊 Métricas do Backtest</div>",
            unsafe_allow_html=True)

cols_m = st.columns(4)
metricas_display = [
    ("Retorno Acumulado", f"{bt['ret_total']:.2%}",
     f"B&H: {mf['Retorno B&H']:.2%}", bt['ret_total'] >= 0),
    ("Sharpe Ratio", f"{bt['sharpe']:.3f}",
     f"Sortino: {mf['Sortino Ratio']:.3f}", bt['sharpe'] >= 1.0),
    ("Win Rate", f"{bt['win_rate']:.1%}",
     f"{bt['n_trades']:,} trades", bt['win_rate'] >= 0.5),
    ("Max Drawdown", f"{bt['max_dd']:.2%}",
     f"Profit Factor: {bt['profit_factor']:.2f}", bt['max_dd'] > -0.20),
]

for col, (label, valor, sub, positivo) in zip(cols_m, metricas_display):
    cor = "delta-pos" if positivo else "delta-neg"
    with col:
        st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>{label}</div>
        <div class='metric-value'>{valor}</div>
        <div class='metric-delta {cor}'>{sub}</div>
        </div>""", unsafe_allow_html=True)

cols_m2 = st.columns(4)
metricas_display2 = [
    ("Capital Final", f"${bt['capital_final']:,.0f}",
     f"Inicial: ${CAPITAL_INICIAL:,.0f}", bt['capital_final'] >= CAPITAL_INICIAL),
    ("Payoff (Ganho/Perda)", f"{mf['Payoff']:.3f}",
     "Gain médio / Loss médio", mf['Payoff'] >= 1.0),
    ("ROC-AUC (Teste)", f"{res['df_res'].iloc[0]['ROC_Test']:.4f}",
     f"Acc: {res['df_res'].iloc[0]['Acc_Test']:.4f}", res['df_res'].iloc[0]['ROC_Test'] >= 0.55),
    ("F1-Score (Teste)", f"{res['df_res'].iloc[0]['F1_Test']:.4f}",
     f"Prec: {res['df_res'].iloc[0]['Prec_Test']:.4f}", res['df_res'].iloc[0]['F1_Test'] >= 0.50),
]

for col, (label, valor, sub, positivo) in zip(cols_m2, metricas_display2):
    cor = "delta-pos" if positivo else "delta-neg"
    with col:
        st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>{label}</div>
        <div class='metric-value'>{valor}</div>
        <div class='metric-delta {cor}'>{sub}</div>
        </div>""", unsafe_allow_html=True)

# ── GRÁFICOS ──────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>📈 Equity Curve & Drawdown</div>",
            unsafe_allow_html=True)

fig_equity = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3], vertical_spacing=0.05)

datas_test = res['datas_test']
eq = bt['equity_curve']
eq_bh = mf['equity_bh']
dd = bt['drawdown']

n_pts = min(len(datas_test), len(eq))
dates_plot = list(datas_test[:n_pts])

fig_equity.add_trace(go.Scatter(
    x=dates_plot, y=eq.values[:n_pts],
    name='Modelo ML', line=dict(color='#58a6ff', width=2),
    fill='tozeroy', fillcolor='rgba(88,166,255,0.05)'
), row=1, col=1)
fig_equity.add_trace(go.Scatter(
    x=dates_plot, y=eq_bh.values[:n_pts],
    name='Buy & Hold', line=dict(color='#d29922', width=1.5, dash='dash')
), row=1, col=1)
fig_equity.add_hline(y=CAPITAL_INICIAL, line_dash='dot', line_color='#30363d', row=1, col=1)

fig_equity.add_trace(go.Scatter(
    x=dates_plot, y=dd.values[:n_pts] * 100,
    name='Drawdown', fill='tozeroy', fillcolor='rgba(248,81,73,0.25)',
    line=dict(color='#f85149', width=1)
), row=2, col=1)

fig_equity.update_layout(
    paper_bgcolor='#0b0e14', plot_bgcolor='#0d1117',
    font=dict(color='#c9d1d9', family='JetBrains Mono'),
    legend=dict(bgcolor='#161b22', bordercolor='#21262d', borderwidth=1),
    height=450, margin=dict(l=10, r=10, t=10, b=10),
    hovermode='x unified',
)
fig_equity.update_xaxes(gridcolor='#21262d', zeroline=False)
fig_equity.update_yaxes(gridcolor='#21262d', zeroline=False)
fig_equity.update_yaxes(title_text='Capital (USD)', row=1, col=1,
                         tickformat='$,.0f')
fig_equity.update_yaxes(title_text='Drawdown (%)', row=2, col=1)
st.plotly_chart(fig_equity, use_container_width=True)

# ── DISTRIBUIÇÃO DAS PROBABILIDADES ──────────────────────────────────────────
col_dist, col_conf = st.columns(2)

with col_dist:
    st.markdown("<div class='section-header'>📉 Distribuição das Probabilidades</div>",
                unsafe_allow_html=True)
    fig_prob = go.Figure()
    fig_prob.add_trace(go.Histogram(
        x=res['y_prob_test'], nbinsx=50, name='Prob. de Alta',
        marker_color='#58a6ff', opacity=0.75
    ))
    fig_prob.add_vline(x=PROB_THRESHOLD, line_color='#3fb950',
                       line_dash='dash', annotation_text=f'Threshold {PROB_THRESHOLD:.0%}')
    fig_prob.add_vline(x=1-PROB_THRESHOLD, line_color='#f85149',
                       line_dash='dash', annotation_text=f'{1-PROB_THRESHOLD:.0%}')
    fig_prob.update_layout(
        paper_bgcolor='#0b0e14', plot_bgcolor='#0d1117',
        font=dict(color='#c9d1d9', family='JetBrains Mono'),
        height=300, margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        xaxis=dict(gridcolor='#21262d', title='Probabilidade'),
        yaxis=dict(gridcolor='#21262d', title='Frequência'),
    )
    st.plotly_chart(fig_prob, use_container_width=True)

with col_conf:
    st.markdown("<div class='section-header'>🔥 Confusion Matrix</div>",
                unsafe_allow_html=True)
    cm = confusion_matrix(res['y_test'], res['y_pred_test'])
    fig_cm = go.Figure(go.Heatmap(
        z=cm, x=['Pred Queda', 'Pred Alta'], y=['Real Queda', 'Real Alta'],
        colorscale=[[0, '#0d1117'], [1, '#1f6feb']],
        text=cm, texttemplate='%{text}',
        textfont=dict(size=16, color='white')
    ))
    fig_cm.update_layout(
        paper_bgcolor='#0b0e14', plot_bgcolor='#0d1117',
        font=dict(color='#c9d1d9', family='JetBrains Mono'),
        height=300, margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# ── RANKING DE MODELOS ────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>🏆 Ranking de Modelos</div>",
            unsafe_allow_html=True)

df_res_disp = res['df_res'].copy()
df_res_disp.index = range(1, len(df_res_disp)+1)
for col_pct in ['ROC_Test','Acc_Test','F1_Test','Prec_Test','Rec_Test']:
    df_res_disp[col_pct] = df_res_disp[col_pct].map('{:.4f}'.format)

st.dataframe(
    df_res_disp[['Modelo','ROC_Test','Acc_Test','F1_Test','Prec_Test','Rec_Test']].rename(columns={
        'ROC_Test': 'ROC-AUC', 'Acc_Test': 'Accuracy',
        'F1_Test': 'F1-Score', 'Prec_Test': 'Precision', 'Rec_Test': 'Recall'
    }),
    use_container_width=True, hide_index=False,
)

# ── HISTÓRICO DE TRADES ───────────────────────────────────────────────────────
if bt['n_trades'] > 0:
    st.markdown("<div class='section-header'>📋 Histórico de Trades</div>",
                unsafe_allow_html=True)
    df_trades = bt['df_trades'].copy()
    df_trades['sinal_str'] = df_trades['sinal'].map({1: '🟢 Compra', -1: '🔴 Venda'})
    df_trades['prob'] = df_trades['prob'].map('{:.3f}'.format)
    df_trades['preco'] = df_trades['preco'].map('{:.4f}'.format)
    df_trades['retorno_real'] = df_trades['retorno_real'].map('{:.4f}'.format)
    df_trades['pnl'] = df_trades['pnl'].map('${:,.2f}'.format)
    df_trades['capital'] = df_trades['capital'].map('${:,.2f}'.format)

    st.dataframe(
        df_trades[['data','sinal_str','prob','preco','retorno_real','pnl','capital']].rename(columns={
            'data':'Data','sinal_str':'Sinal','prob':'Prob.','preco':'Preço',
            'retorno_real':'Ret. Real','pnl':'P&L','capital':'Capital'
        }).tail(200),
        use_container_width=True, height=300,
    )
else:
    st.markdown("""
    <div class='warning-box'>
    ⚠️ Nenhum trade gerado com o threshold atual. Considere reduzir o valor de <strong>Prob. mínima para operar</strong> na sidebar.
    </div>
    """, unsafe_allow_html=True)

# ── SUMÁRIO DOS PARÂMETROS USADOS ─────────────────────────────────────────────
with st.expander("📋 Parâmetros usados neste backtest"):
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.markdown("**Ativo & Dados**")
        st.write(f"- Ticker: `{TICKER}`")
        st.write(f"- Histórico: `{ANOS_HISTORICO} anos`")
        st.write(f"- Timeframe: `{TIMEFRAME}`")
        st.write(f"- Split Treino/Val/Teste: `{TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%}`")
        st.write(f"- Max Features: `{MAX_FEATURES}`")
        st.write(f"- Correlação máx.: `{CORRELACAO_MAX}`")
    with col_p2:
        st.markdown("**Gestão de Risco**")
        st.write(f"- Capital inicial: `${CAPITAL_INICIAL:,.0f}`")
        st.write(f"- Risco por trade: `{RISCO_POR_TRADE:.2%}`")
        st.write(f"- Stop Loss: `{STOP_LOSS_PCT:.2%}`")
        st.write(f"- Take Profit: `{TAKE_PROFIT_PCT:.2%}`")
        st.write(f"- R/R Ratio: `1:{RR_RATIO:.1f}`")
        st.write(f"- Threshold: `{PROB_THRESHOLD:.2f}`")
    with col_p3:
        st.markdown("**Custos & Indicadores**")
        st.write(f"- Spread: `{SPREAD_PIPS:.4f}`")
        st.write(f"- Slippage: `{SLIPPAGE:.4f}`")
        st.write(f"- Custo operacional: `{CUSTO_OPERACIONAL:.4f}`")
        st.write(f"- ADX limiar tendência: `{ADX_TENDENCIA_THRESH}`")
        st.write(f"- Regime vol. mult.: `{REGIME_VOL_THRESH}`")
        st.write(f"- RSI sobrecomprado: `{RSI_SOBRECOMPRADO}` / sobrevendido: `{RSI_SOBREVENDIDO}`")

# ── DISCLAIMER ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class='warning-box' style='text-align:center;'>
⚠️ <strong>DISCLAIMER:</strong> Este dashboard é exclusivamente para fins educacionais e de pesquisa quantitativa. 
Resultados passados não garantem resultados futuros. NÃO constitui recomendação de investimento. 
Consulte sempre um profissional certificado antes de tomar decisões financeiras.
</div>
""", unsafe_allow_html=True)
