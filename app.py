# -*- coding: utf-8 -*-
# ==============================================================================
# SISTEMA QUANTITATIVO DE ML PARA TRADING — STREAMLIT DASHBOARD
# Versão: 4.0 | Dark Mode Profissional
# ==============================================================================

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import time
import os

# ── Streamlit config (DEVE ser a 1ª chamada) ──────────────────────────────────
st.set_page_config(
    page_title="QuantTrading ML Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Dark Mode Profissional ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg-main:    #0b0e14;
    --bg-card:    #131720;
    --bg-input:   #1a1f2e;
    --border:     #252d3d;
    --accent:     #58a6ff;
    --success:    #3fb950;
    --danger:     #f85149;
    --warning:    #d29922;
    --neutral:    #8b949e;
    --text-main:  #f0f6fc;
    --text-sub:   #8b949e;
    --highlight:  #bc8cff;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-main) !important;
    font-family: 'Syne', sans-serif;
    color: var(--text-main);
}

[data-testid="stSidebar"] {
    background-color: #0d1117 !important;
    border-right: 1px solid var(--border);
}

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--accent); }
.metric-label {
    font-size: 11px;
    color: var(--text-sub);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
    font-family: 'Space Mono', monospace;
}
.metric-value {
    font-size: 26px;
    font-weight: 800;
    font-family: 'Space Mono', monospace;
}
.metric-value.positive { color: var(--success); }
.metric-value.negative { color: var(--danger); }
.metric-value.neutral  { color: var(--accent);  }
.metric-value.warning  { color: var(--warning); }

.signal-badge {
    display: inline-block;
    padding: 10px 24px;
    border-radius: 8px;
    font-size: 18px;
    font-weight: 800;
    letter-spacing: 2px;
    font-family: 'Space Mono', monospace;
}
.signal-buy     { background: rgba(63,185,80,0.15); color: #3fb950; border: 1px solid #3fb950; }
.signal-sell    { background: rgba(248,81,73,0.15);  color: #f85149; border: 1px solid #f85149; }
.signal-neutral { background: rgba(139,148,158,0.15);color: #8b949e; border: 1px solid #8b949e; }

.section-title {
    font-size: 13px;
    color: var(--text-sub);
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin-bottom: 16px;
    font-family: 'Space Mono', monospace;
}

.info-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    margin: 8px 0;
}

.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
.status-dot.green { background: var(--success); }
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}

/* Ocultar itens padrão do Streamlit */
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
div[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# ── IMPORTS DE ML (com graceful fallback) ─────────────────────────────────────
# ==============================================================================
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from scipy import stats

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
    from catboost import CatBoostClassifier
    CAT_AVAILABLE = True
except ImportError:
    CAT_AVAILABLE = False

try:
    import ta
    from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
    from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator
    from ta.volatility import AverageTrueRange, BollingerBands
    from ta.volume import OnBalanceVolumeIndicator
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False


# ==============================================================================
# ── FUNÇÕES DO PIPELINE DE ML ─────────────────────────────────────────────────
# ==============================================================================

def baixar_dados(ticker: str, anos: int = 5, interval: str = "1d") -> pd.DataFrame:
    data_inicio = (datetime.now() - timedelta(days=anos * 365)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=data_inicio, interval=interval,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().capitalize() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    if 'Volume' not in df.columns:
        df['Volume'] = 0
    return df


def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()
    df = df[~df.index.duplicated(keep='last')]
    df = df[(df['Close'] > 0) & (df['Open'] > 0)]
    df['Volume'] = df['Volume'].fillna(0).clip(lower=0)
    df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df


def construir_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['retorno_atual']     = df['Close'].pct_change()
    df['retorno_futuro_pct'] = df['Close'].pct_change().shift(-1) * 100
    df['target']            = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df


def engenharia_de_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, o, v = df['Close'], df['High'], df['Low'], df['Open'], df['Volume']
    ret = c.pct_change()

    # Retornos
    for p in [1, 2, 3, 5, 10, 20]:
        df[f'ret_{p}d'] = c.pct_change(p)

    # Médias móveis
    for p in [5, 10, 20, 50, 100, 200]:
        df[f'sma_{p}'] = c.rolling(p).mean()
    df['sma_cross_5_20']   = (df['sma_5']  > df['sma_20']).astype(int)
    df['sma_cross_20_50']  = (df['sma_20'] > df['sma_50']).astype(int)
    df['sma_cross_50_200'] = (df['sma_50'] > df['sma_200']).astype(int)
    for p in [5, 10, 20, 50]:
        df[f'ema_{p}']    = c.ewm(span=p).mean()
        df[f'dist_sma{p}'] = (c - df[f'sma_{p}']) / df[f'sma_{p}'].clip(lower=1e-9)

    # Volatilidade
    for p in [5, 10, 20, 60]:
        df[f'vol_{p}'] = ret.rolling(p).std() * np.sqrt(252)
    df['vol_regime']   = df['vol_20'] / df['vol_60'].clip(lower=1e-9)
    df['parkinson_vol'] = np.sqrt(1/(4*np.log(2)) * np.log(h/l.clip(lower=1e-9))**2)

    # RSI
    for p in [7, 14, 21]:
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(p).mean()
        loss = (-delta.clip(upper=0)).rolling(p).mean()
        rs = gain / loss.clip(lower=1e-9)
        df[f'rsi_{p}'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    df['macd']        = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']

    # Bollinger Bands
    for p in [20]:
        mid = c.rolling(p).mean()
        std = c.rolling(p).std()
        df[f'bb_upper_{p}'] = mid + 2*std
        df[f'bb_lower_{p}'] = mid - 2*std
        df[f'bb_pos_{p}']   = (c - df[f'bb_lower_{p}']) / (df[f'bb_upper_{p}'] - df[f'bb_lower_{p}']).clip(lower=1e-9)
        df[f'bb_width_{p}'] = (df[f'bb_upper_{p}'] - df[f'bb_lower_{p}']) / mid.clip(lower=1e-9)

    # ATR
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    for p in [14, 20]:
        df[f'atr_{p}']     = tr.rolling(p).mean()
        df[f'atr_pct_{p}'] = df[f'atr_{p}'] / c.clip(lower=1e-9)

    # ADX
    df['adx_14'] = 0.0
    if TA_AVAILABLE:
        try:
            adx = ADXIndicator(h, l, c, window=14)
            df['adx_14'] = adx.adx()
        except Exception:
            pass

    # Estocástico
    low14  = l.rolling(14).min()
    high14 = h.rolling(14).max()
    df['stoch_k'] = 100 * (c - low14) / (high14 - low14).clip(lower=1e-9)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # Volume
    df['obv']        = (np.sign(ret) * v).cumsum()
    df['vma_20']     = v.rolling(20).mean()
    df['vol_ratio']  = v / df['vma_20'].clip(lower=1e-9)

    # Candlestick
    df['corpo']      = (c - o).abs()
    df['corpo_pct']  = df['corpo'] / o.clip(lower=1e-9)
    df['range']      = h - l
    df['range_pct']  = df['range'] / o.clip(lower=1e-9)
    df['gap']        = (o - c.shift(1)) / c.shift(1).clip(lower=1e-9)
    df['direcao_candle'] = np.sign(c - o)

    # Lags
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f'ret_lag_{lag}'] = ret.shift(lag)
    for lag in [1, 3, 5]:
        df[f'rsi14_lag_{lag}'] = df['rsi_14'].shift(lag)
    for lag in [1, 3]:
        df[f'macd_lag_{lag}'] = df['macd'].shift(lag)

    # Rolling stats
    for p in [20, 60]:
        df[f'skew_{p}'] = ret.rolling(p).skew()
        df[f'kurt_{p}'] = ret.rolling(p).kurt()

    # Regimes
    df['regime_tendencia']  = (df['adx_14'] > 25).astype(int)
    df['regime_alta_vol']   = (df['vol_regime'] > 1.2).astype(int)
    df['regime_bull']       = ((c > df['sma_50']) & (c > df['sma_200'])).astype(int)
    df['regime_bear']       = ((c < df['sma_50']) & (c < df['sma_200'])).astype(int)

    # Sazonalidade
    df['dia_semana'] = df.index.dayofweek
    df['mes']        = df.index.month
    df['trimestre']  = df.index.quarter
    df['fim_mes']    = (df.index.day >= 25).astype(int)
    df['inicio_mes'] = (df.index.day <= 5).astype(int)

    return df


COLUNAS_NAO_FEATURE = [
    'Open','High','Low','Close','Volume',
    'target','retorno_futuro_pct','retorno_atual',
]

def selecionar_features(df: pd.DataFrame, max_features: int = 50) -> list:
    features_candidatas = [c for c in df.columns if c not in COLUNAS_NAO_FEATURE]
    df_feats = df[features_candidatas + ['target']].copy()
    nan_pct = df_feats.isnull().mean()
    features_ok = [f for f in nan_pct[nan_pct < 0.30].index if f != 'target']
    df_clean = df_feats[features_ok + ['target']].dropna()
    if len(df_clean) < 50 or len(features_ok) == 0:
        return features_ok[:max_features]
    variancias = df_clean[features_ok].var()
    features_ok = variancias[variancias > 1e-10].index.tolist()
    corr_matrix = df_clean[features_ok].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    remover = {col for col in upper.columns if any(upper[col] > 0.95)}
    features_ok = [f for f in features_ok if f not in remover]
    X = df_clean[features_ok].values
    y = df_clean['target'].values
    mi = mutual_info_classif(X, y, random_state=42)
    ranking = pd.Series(mi, index=features_ok).sort_values(ascending=False)
    return ranking.head(max_features).index.tolist()


def split_temporal(X, y, datas, closes, retornos,
                   train_r=0.70, val_r=0.15):
    n = len(X)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)
    return {
        'X_train': X[:n_train], 'y_train': y[:n_train],
        'X_val':   X[n_train:n_train+n_val], 'y_val': y[n_train:n_train+n_val],
        'X_test':  X[n_train+n_val:],        'y_test': y[n_train+n_val:],
        'datas_train': datas[:n_train],
        'datas_val':   datas[n_train:n_train+n_val],
        'datas_test':  datas[n_train+n_val:],
        'closes_test': closes[n_train+n_val:],
        'retornos_test': retornos[n_train+n_val:],
        'n_train': n_train, 'n_val': n_val, 'n_test': n - n_train - n_val,
    }


def treinar_modelos(X_train, y_train, X_val, y_val):
    modelos = {}
    modelos['LogReg'] = LogisticRegression(C=0.1, max_iter=500, random_state=42)
    modelos['RandForest'] = RandomForestClassifier(n_estimators=100, max_depth=6,
                                                    min_samples_leaf=10, random_state=42, n_jobs=-1)
    if XGB_AVAILABLE:
        modelos['XGBoost'] = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                            eval_metric='logloss', random_state=42, n_jobs=-1,
                                            verbosity=0)
    if LGB_AVAILABLE:
        modelos['LightGBM'] = LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                              num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
    if CAT_AVAILABLE:
        modelos['CatBoost'] = CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05,
                                                   random_state=42, verbose=0)
    resultados = {}
    for nome, modelo in modelos.items():
        try:
            modelo.fit(X_train, y_train)
            prob_val = modelo.predict_proba(X_val)[:, 1]
            auc_val  = roc_auc_score(y_val, prob_val)
            resultados[nome] = {'modelo': modelo, 'auc_val': auc_val}
        except Exception as e:
            pass

    # Voting ensemble (se ≥ 2 modelos)
    if len(resultados) >= 2:
        estimadores = [(n, i['modelo']) for n, i in resultados.items()][:4]
        try:
            voting = VotingClassifier(estimators=estimadores, voting='soft', n_jobs=-1)
            voting.fit(X_train, y_train)
            prob_v = voting.predict_proba(X_val)[:, 1]
            auc_v  = roc_auc_score(y_val, prob_v)
            resultados['Ensemble'] = {'modelo': voting, 'auc_val': auc_v}
        except Exception:
            pass

    return resultados


def metricas_financeiras(retornos_modelo, retornos_bh, capital_inicial=100_000):
    rm = pd.Series(retornos_modelo)
    rb = pd.Series(retornos_bh)
    equity    = capital_inicial * (1 + rm).cumprod()
    equity_bh = capital_inicial * (1 + rb).cumprod()
    ret_total = equity.iloc[-1] / capital_inicial - 1 if len(equity) > 0 else 0
    ret_bh    = equity_bh.iloc[-1] / capital_inicial - 1 if len(equity_bh) > 0 else 0
    std = rm.std()
    sharpe = rm.mean() / std * np.sqrt(252) if std > 0 else 0
    downside = rm[rm < 0].std()
    sortino = rm.mean() / downside * np.sqrt(252) if downside > 0 else 0
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    max_dd = drawdown.min()
    wins = int((rm > 0).sum())
    losses = int((rm <= 0).sum())
    n = len(rm.dropna())
    win_rate = wins / n if n > 0 else 0
    gross_profit = rm[rm > 0].sum()
    gross_loss   = abs(rm[rm < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    return {
        'ret_total': ret_total, 'ret_bh': ret_bh,
        'sharpe': sharpe, 'sortino': sortino,
        'max_dd': max_dd, 'win_rate': win_rate,
        'profit_factor': profit_factor,
        'equity': equity, 'equity_bh': equity_bh, 'drawdown': drawdown,
    }


def backtest_realista(y_pred, y_prob, closes, retornos, datas,
                       capital_inicial=100_000, prob_threshold=0.55,
                       risco_trade=0.02, stop_loss=0.02, take_profit=0.04,
                       spread=0.0002, slippage=0.0001):
    capital = capital_inicial
    equity_curve = [capital]
    trades = []
    for i in range(len(y_pred)):
        ret_real = retornos[i] / 100
        sinal = 0
        if y_prob[i] > prob_threshold:
            sinal = 1
        elif y_prob[i] < (1 - prob_threshold):
            sinal = -1
        if sinal == 0:
            equity_curve.append(capital)
            continue
        custo_entrada = (spread + slippage) * capital
        tamanho = capital * risco_trade
        pnl = tamanho * sinal * ret_real
        if sinal == 1:
            if ret_real < -stop_loss:
                pnl = -tamanho * stop_loss
            elif ret_real > take_profit:
                pnl = tamanho * take_profit
        else:
            if ret_real > stop_loss:
                pnl = -tamanho * stop_loss
            elif ret_real < -take_profit:
                pnl = tamanho * take_profit
        capital += pnl - custo_entrada
        equity_curve.append(capital)
        trades.append({'data': datas[i], 'sinal': sinal, 'prob': y_prob[i],
                       'preco': closes[i], 'ret_real': ret_real, 'pnl': pnl - custo_entrada})

    df_t = pd.DataFrame(trades)
    n_t  = len(df_t)
    eq   = pd.Series(equity_curve)
    roll_max = eq.cummax()
    dd = (eq - roll_max) / roll_max
    win_rate = (df_t['pnl'] > 0).sum() / n_t if n_t > 0 else 0
    pf_num = df_t[df_t['pnl'] > 0]['pnl'].sum() if n_t > 0 else 0
    pf_den = abs(df_t[df_t['pnl'] < 0]['pnl'].sum()) if n_t > 0 else 1
    pf     = pf_num / pf_den if pf_den > 0 else float('inf')
    return {
        'equity_curve': eq, 'df_trades': df_t, 'n_trades': n_t,
        'ret_total': capital / capital_inicial - 1,
        'win_rate': win_rate, 'profit_factor': pf,
        'max_dd': dd.min(), 'capital_final': capital,
    }


def predict_next_day(ticker, modelo, scaler_obj, feature_names, threshold=0.55):
    df_novo = baixar_dados(ticker, anos=1)
    df_novo = limpar_dados(df_novo)
    df_novo = construir_target(df_novo)
    df_novo = engenharia_de_features(df_novo)
    for f in feature_names:
        if f not in df_novo.columns:
            df_novo[f] = 0.0
    ultima = df_novo[feature_names].iloc[-1:].fillna(0)
    X_prod = scaler_obj.transform(ultima.values)
    prob_alta = modelo.predict_proba(X_prod)[0, 1]
    prob_queda = 1 - prob_alta
    if prob_alta >= threshold:
        sinal = 'BUY'
    elif prob_queda >= threshold:
        sinal = 'SELL'
    else:
        sinal = 'NEUTRAL'
    return {
        'sinal': sinal,
        'prob_alta': prob_alta, 'prob_queda': prob_queda,
        'confianca': max(prob_alta, prob_queda),
        'preco_atual': df_novo['Close'].iloc[-1],
        'rsi': df_novo['rsi_14'].iloc[-1] if 'rsi_14' in df_novo.columns else 50,
        'data_ref': df_novo.index[-1].date(),
    }


# ==============================================================================
# ── CACHE: pipeline completo ──────────────────────────────────────────────────
# ==============================================================================

@st.cache_data(ttl=60, show_spinner=False)
def rodar_pipeline(ticker, anos, threshold, capital_inicial,
                   risco_trade, stop_loss, take_profit):
    resultado = {}
    try:
        df_raw = baixar_dados(ticker, anos)
        df     = limpar_dados(df_raw)
        df     = construir_target(df)
        df     = engenharia_de_features(df)

        features = selecionar_features(df)
        if len(features) < 5:
            return {'erro': 'Features insuficientes. Tente outro ativo ou período maior.'}

        colunas = features + ['target', 'Close', 'retorno_futuro_pct', 'retorno_atual']
        colunas = [c for c in colunas if c in df.columns]
        df_mod  = df[colunas].dropna()

        if len(df_mod) < 200:
            return {'erro': 'Dados insuficientes para treino (mínimo 200 barras).'}

        X  = df_mod[features].values
        y  = df_mod['target'].values
        datas    = df_mod.index
        closes   = df_mod['Close'].values
        retornos = df_mod['retorno_futuro_pct'].values

        splits = split_temporal(X, y, datas, closes, retornos)

        scaler    = RobustScaler()
        X_train_s = scaler.fit_transform(splits['X_train'])
        X_val_s   = scaler.transform(splits['X_val'])
        X_test_s  = scaler.transform(splits['X_test'])

        modelos_treinados = treinar_modelos(X_train_s, splits['y_train'],
                                             X_val_s,   splits['y_val'])
        if not modelos_treinados:
            return {'erro': 'Nenhum modelo treinou com sucesso.'}

        melhor_nome, melhor_info = max(modelos_treinados.items(),
                                        key=lambda x: x[1]['auc_val'])
        melhor_modelo = melhor_info['modelo']

        # Avaliação no teste
        y_pred = melhor_modelo.predict(X_test_s)
        y_prob = melhor_modelo.predict_proba(X_test_s)[:, 1]
        roc_test = roc_auc_score(splits['y_test'], y_prob)
        acc_test = accuracy_score(splits['y_test'], y_pred)
        f1_test  = f1_score(splits['y_test'], y_pred, zero_division=0)

        sinais     = np.where(y_pred == 1, 1, -1)
        ret_modelo = pd.Series(sinais * splits['retornos_test'] / 100,
                               index=splits['datas_test'])
        ret_bh     = pd.Series(splits['retornos_test'] / 100,
                               index=splits['datas_test'])
        mf = metricas_financeiras(ret_modelo, ret_bh, capital_inicial)

        backtest = backtest_realista(
            y_pred, y_prob,
            splits['closes_test'], splits['retornos_test'], splits['datas_test'],
            capital_inicial, threshold, risco_trade, stop_loss, take_profit,
        )

        # Previsão próximo dia
        previsao = predict_next_day(ticker, melhor_modelo, scaler, features, threshold)

        # Ranking modelos
        ranking = pd.DataFrame([
            {'Modelo': n, 'ROC-AUC Val': f"{i['auc_val']:.4f}"}
            for n, i in sorted(modelos_treinados.items(),
                                key=lambda x: x[1]['auc_val'], reverse=True)
        ])

        resultado = {
            'ticker': ticker, 'df_raw': df_raw, 'splits': splits,
            'melhor_nome': melhor_nome, 'roc_test': roc_test,
            'acc_test': acc_test, 'f1_test': f1_test,
            'metricas_fin': mf, 'backtest': backtest,
            'previsao': previsao, 'ranking': ranking,
            'features': features, 'y_pred': y_pred, 'y_prob': y_prob,
            'datas_test': splits['datas_test'],
            'closes_test': splits['closes_test'],
        }
    except Exception as e:
        resultado = {'erro': str(e)}
    return resultado


# ==============================================================================
# ── SIDEBAR ───────────────────────────────────────────────────────────────────
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <div style='font-size:28px;'>📈</div>
        <div style='font-size:16px; font-weight:800; color:#f0f6fc; font-family:Syne;'>QuantTrading ML</div>
        <div style='font-size:10px; color:#8b949e; letter-spacing:2px; font-family:Space Mono;'>SISTEMA QUANT v4.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">⚙ Configuração</div>', unsafe_allow_html=True)

    ATIVOS_POPULARES = {
        "USD/JPY": "USDJPY=X", "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X",
        "Ouro":    "GC=F",     "Petróleo": "CL=F",    "S&P 500": "^GSPC",
        "NASDAQ":  "^IXIC",    "Bovespa": "^BVSP",    "Bitcoin": "BTC-USD",
        "Ethereum":"ETH-USD",  "PETR4":   "PETR4.SA", "VALE3":   "VALE3.SA",
        "AAPL": "AAPL",        "NVDA": "NVDA",         "TSLA": "TSLA",
    }
    ativo_sel = st.selectbox("Ativo", list(ATIVOS_POPULARES.keys()), index=0)
    ticker_custom = st.text_input("Ou digite o ticker:", placeholder="Ex: MSFT, XAUUSD=X")
    TICKER = ticker_custom.upper() if ticker_custom else ATIVOS_POPULARES[ativo_sel]

    anos = st.slider("Histórico (anos)", 2, 15, 5)

    st.markdown('<div class="section-title" style="margin-top:16px">🎯 Parâmetros do Modelo</div>',
                unsafe_allow_html=True)
    threshold   = st.slider("Threshold de Probabilidade", 0.50, 0.90, 0.55, 0.01,
                             help="Mínimo de confiança para emitir sinal")
    capital     = st.number_input("Capital Inicial (USD)", 1_000, 10_000_000, 100_000, step=1_000)
    risco_trade = st.slider("Risco por Trade (%)", 0.5, 5.0, 2.0, 0.5) / 100
    stop_loss   = st.slider("Stop-Loss (%)", 0.5, 10.0, 2.0, 0.5) / 100
    take_profit = st.slider("Take-Profit (%)", 1.0, 20.0, 4.0, 0.5) / 100

    st.markdown("<br>", unsafe_allow_html=True)
    executar = st.button("🚀 Executar Backtest", type="primary", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    auto_refresh = st.checkbox("Auto-refresh (60s)", value=False)
    if auto_refresh:
        st.markdown('<div style="font-size:11px; color:#3fb950;"><span class="status-dot green"></span>Atualização automática ativa</div>',
                    unsafe_allow_html=True)
        time.sleep(1)
        st.rerun()

    st.markdown("""
    <div style='margin-top:24px; font-size:10px; color:#555; line-height:1.6;'>
    ⚠ Fins educacionais apenas.<br>
    Não constitui recomendação<br>de investimento.
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# ── HEADER ────────────────────────────────────────────────────────────────────
# ==============================================================================
st.markdown(f"""
<div style='padding: 8px 0 24px 0;'>
    <div style='font-size:11px; color:#8b949e; letter-spacing:3px; font-family:Space Mono;'>
        SISTEMA QUANTITATIVO DE ML PARA TRADING
    </div>
    <div style='font-size:32px; font-weight:800; color:#f0f6fc; line-height:1.2; margin-top:4px;'>
        {TICKER}
        <span style='font-size:14px; font-weight:400; color:#8b949e; margin-left:12px;'>
            Dashboard de Análise
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# ── ESTADO DA SESSÃO ──────────────────────────────────────────────────────────
# ==============================================================================
if 'resultado' not in st.session_state:
    st.session_state['resultado'] = None
if 'ultimo_ticker' not in st.session_state:
    st.session_state['ultimo_ticker'] = ''

if executar or (st.session_state['ultimo_ticker'] != TICKER and st.session_state['resultado'] is None):
    with st.spinner(f'⚙ Processando pipeline ML para **{TICKER}**... pode levar 1-3 minutos.'):
        res = rodar_pipeline(TICKER, anos, threshold, capital,
                             risco_trade, stop_loss, take_profit)
        st.session_state['resultado'] = res
        st.session_state['ultimo_ticker'] = TICKER

res = st.session_state.get('resultado')

# ==============================================================================
# ── ESTADO INICIAL (sem execução) ─────────────────────────────────────────────
# ==============================================================================
if res is None:
    # Mostrar preview com dados de preço
    @st.cache_data(ttl=300)
    def carregar_preview(ticker):
        try:
            df = yf.download(ticker, period="1y", interval="1d",
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [str(c).strip().capitalize() for c in df.columns]
            return df
        except Exception:
            return pd.DataFrame()

    df_prev = carregar_preview(TICKER)

    if not df_prev.empty and 'Close' in df_prev.columns:
        preco_atual  = float(df_prev['Close'].iloc[-1])
        preco_ontem  = float(df_prev['Close'].iloc[-2])
        var_diaria   = (preco_atual - preco_ontem) / preco_ontem * 100
        var_anual    = (preco_atual / float(df_prev['Close'].iloc[0]) - 1) * 100
        max_52 = float(df_prev['Close'].max())
        min_52 = float(df_prev['Close'].min())

        c1, c2, c3, c4 = st.columns(4)
        cor_dia   = "positive" if var_diaria >= 0 else "negative"
        cor_anual = "positive" if var_anual  >= 0 else "negative"
        sinal_dia   = "+" if var_diaria >= 0 else ""
        sinal_anual = "+" if var_anual  >= 0 else ""

        c1.markdown(f"""<div class="metric-card">
            <div class="metric-label">Preço Atual</div>
            <div class="metric-value neutral">{preco_atual:,.4f}</div>
        </div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="metric-card">
            <div class="metric-label">Variação Diária</div>
            <div class="metric-value {cor_dia}">{sinal_dia}{var_diaria:.2f}%</div>
        </div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class="metric-card">
            <div class="metric-label">Retorno 1 Ano</div>
            <div class="metric-value {cor_anual}">{sinal_anual}{var_anual:.1f}%</div>
        </div>""", unsafe_allow_html=True)
        c4.markdown(f"""<div class="metric-card">
            <div class="metric-label">Range 52 Semanas</div>
            <div class="metric-value neutral">{min_52:,.2f} – {max_52:,.2f}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Gráfico de preço
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_prev.index, y=df_prev['Close'],
            mode='lines', name='Close',
            line=dict(color='#58a6ff', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(88,166,255,0.05)',
        ))
        fig.update_layout(
            title=f'{TICKER} — Histórico 1 Ano',
            paper_bgcolor='#0b0e14', plot_bgcolor='#131720',
            font=dict(color='#f0f6fc', family='Space Mono'),
            xaxis=dict(gridcolor='#252d3d', showgrid=True),
            yaxis=dict(gridcolor='#252d3d', showgrid=True),
            height=350, margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="info-box" style="text-align:center; padding:32px;">
        <div style="font-size:32px; margin-bottom:12px;">🚀</div>
        <div style="font-size:16px; font-weight:700; color:#f0f6fc;">Pronto para Executar o Backtest</div>
        <div style="font-size:13px; color:#8b949e; margin-top:8px;">
            Configure os parâmetros na barra lateral e clique em <strong>Executar Backtest</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ==============================================================================
# ── ERRO ──────────────────────────────────────────────────────────────────────
# ==============================================================================
if 'erro' in res:
    st.error(f"❌ **Erro no pipeline:** {res['erro']}")
    st.stop()


# ==============================================================================
# ── RESULTADOS ────────────────────────────────────────────────────────────────
# ==============================================================================
mf       = res['metricas_fin']
bt       = res['backtest']
previsao = res['previsao']

# ── 1. SINAL DO DIA ───────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📡 Sinal para o Próximo Dia</div>', unsafe_allow_html=True)

sinal_map = {
    'BUY':     ('signal-buy',     '▲ COMPRA',  f"Prob. Alta: {previsao['prob_alta']:.1%}"),
    'SELL':    ('signal-sell',    '▼ VENDA',   f"Prob. Queda: {previsao['prob_queda']:.1%}"),
    'NEUTRAL': ('signal-neutral', '● NEUTRO',  f"Confiança insuficiente ({previsao['confianca']:.1%})"),
}
cls, label, desc = sinal_map[previsao['sinal']]

col_sinal, col_preco, col_rsi, col_conf, col_data = st.columns([2, 1.5, 1.5, 1.5, 1.5])
col_sinal.markdown(f"""
<div class="metric-card" style="padding:24px; text-align:center;">
    <div class="metric-label">Sinal ML</div>
    <div style="margin-top:8px">
        <span class="signal-badge {cls}">{label}</span>
    </div>
    <div style="font-size:11px; color:#8b949e; margin-top:10px;">{desc}</div>
</div>
""", unsafe_allow_html=True)

col_preco.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Preço Atual</div>
    <div class="metric-value neutral">{previsao['preco_atual']:,.4f}</div>
    <div style="font-size:10px; color:#555; margin-top:4px;">Ref: {previsao['data_ref']}</div>
</div>
""", unsafe_allow_html=True)

rsi_val  = previsao['rsi']
rsi_cor  = "danger" if rsi_val > 70 else ("positive" if rsi_val < 30 else "neutral")
col_rsi.markdown(f"""
<div class="metric-card">
    <div class="metric-label">RSI (14)</div>
    <div class="metric-value {rsi_cor}">{rsi_val:.1f}</div>
    <div style="font-size:10px; color:#555; margin-top:4px;">
        {'Sobrecomprado' if rsi_val > 70 else ('Sobrevendido' if rsi_val < 30 else 'Neutro')}
    </div>
</div>
""", unsafe_allow_html=True)

conf_cor = "positive" if previsao['confianca'] >= threshold else "warning"
col_conf.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Confiança</div>
    <div class="metric-value {conf_cor}">{previsao['confianca']:.1%}</div>
    <div style="font-size:10px; color:#555; margin-top:4px;">Threshold: {threshold:.0%}</div>
</div>
""", unsafe_allow_html=True)

col_data.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Modelo</div>
    <div class="metric-value neutral" style="font-size:16px;">{res['melhor_nome']}</div>
    <div style="font-size:10px; color:#555; margin-top:4px;">ROC-AUC: {res['roc_test']:.4f}</div>
</div>
""", unsafe_allow_html=True)


# ── 2. MÉTRICAS PRINCIPAIS ────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title">📊 Performance do Backtest</div>', unsafe_allow_html=True)

m1, m2, m3, m4, m5, m6 = st.columns(6)

ret_cor = "positive" if mf['ret_total'] >= 0 else "negative"
bh_cor  = "positive" if mf['ret_bh']    >= 0 else "negative"

def fmt_pct(v): return f"{v:+.1%}"
def fmt_num(v, decimals=2): return f"{v:.{decimals}f}"

for col, label, val, cor in [
    (m1, "Retorno Modelo", fmt_pct(mf['ret_total']), ret_cor),
    (m2, "Retorno B&H",    fmt_pct(mf['ret_bh']),    bh_cor),
    (m3, "Sharpe Ratio",   fmt_num(mf['sharpe']),    "neutral"),
    (m4, "Win Rate",       f"{mf['win_rate']:.1%}",  "positive" if mf['win_rate'] > 0.5 else "warning"),
    (m5, "Max Drawdown",   fmt_pct(mf['max_dd']),    "negative"),
    (m6, "Profit Factor",  fmt_num(mf['profit_factor']) if mf['profit_factor'] != float('inf') else "∞", "neutral"),
]:
    col.markdown(f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {cor}">{val}</div>
    </div>""", unsafe_allow_html=True)


# ── 3. EQUITY CURVES ──────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["📈 Equity Curve", "📉 Drawdown", "🎯 Distribuição de Sinais", "🏆 Ranking de Modelos"])

with tab1:
    datas_test = res['datas_test']
    eq = mf['equity']
    eq_bh = mf['equity_bh']
    n = min(len(datas_test), len(eq), len(eq_bh))

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=list(datas_test[:n]), y=list(eq[:n]),
        mode='lines', name='Modelo ML',
        line=dict(color='#58a6ff', width=2),
    ))
    fig_eq.add_trace(go.Scatter(
        x=list(datas_test[:n]), y=list(eq_bh[:n]),
        mode='lines', name='Buy & Hold',
        line=dict(color='#8b949e', width=1.5, dash='dot'),
    ))
    fig_eq.update_layout(
        title='Equity Curve — Modelo vs Buy & Hold',
        paper_bgcolor='#0b0e14', plot_bgcolor='#131720',
        font=dict(color='#f0f6fc', family='Space Mono', size=11),
        xaxis=dict(gridcolor='#252d3d'),
        yaxis=dict(gridcolor='#252d3d', tickprefix='$', tickformat=',.0f'),
        legend=dict(bgcolor='#131720', bordercolor='#252d3d'),
        height=400, margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_eq, use_container_width=True)

with tab2:
    dd = mf['drawdown']
    n  = min(len(datas_test), len(dd))
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=list(datas_test[:n]), y=list(dd[:n] * 100),
        mode='lines', name='Drawdown',
        fill='tozeroy',
        line=dict(color='#f85149', width=1.5),
        fillcolor='rgba(248,81,73,0.15)',
    ))
    fig_dd.update_layout(
        title='Drawdown (%)',
        paper_bgcolor='#0b0e14', plot_bgcolor='#131720',
        font=dict(color='#f0f6fc', family='Space Mono', size=11),
        xaxis=dict(gridcolor='#252d3d'),
        yaxis=dict(gridcolor='#252d3d', ticksuffix='%'),
        height=350, margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_dd, use_container_width=True)

with tab3:
    y_prob = res['y_prob']
    y_pred = res['y_pred']
    fig_dist = go.Figure()
    prob_long  = y_prob[y_pred == 1]
    prob_short = y_prob[y_pred == 0]
    fig_dist.add_trace(go.Histogram(x=prob_long,  nbinsx=40, name='Sinal COMPRA',
                                    marker_color='rgba(63,185,80,0.6)'))
    fig_dist.add_trace(go.Histogram(x=prob_short, nbinsx=40, name='Sinal VENDA',
                                    marker_color='rgba(248,81,73,0.6)'))
    fig_dist.add_vline(x=threshold, line_dash='dash', line_color='#d29922',
                       annotation_text=f'Threshold {threshold:.0%}',
                       annotation_font_color='#d29922')
    fig_dist.update_layout(
        title='Distribuição de Probabilidades por Sinal',
        barmode='overlay',
        paper_bgcolor='#0b0e14', plot_bgcolor='#131720',
        font=dict(color='#f0f6fc', family='Space Mono', size=11),
        xaxis=dict(gridcolor='#252d3d', title='Probabilidade de Alta'),
        yaxis=dict(gridcolor='#252d3d', title='Frequência'),
        legend=dict(bgcolor='#131720', bordercolor='#252d3d'),
        height=350, margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with tab4:
    ranking = res['ranking']
    st.dataframe(
        ranking,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Modelo":    st.column_config.TextColumn("Modelo", width="medium"),
            "ROC-AUC Val": st.column_config.TextColumn("ROC-AUC (Validação)", width="medium"),
        },
    )
    st.markdown(f"""
    <div class="info-box" style="margin-top:12px;">
        <b style="color:#58a6ff">Melhor Modelo:</b> {res['melhor_nome']}<br>
        <b style="color:#58a6ff">ROC-AUC Teste:</b> {res['roc_test']:.4f} &nbsp;|&nbsp;
        <b style="color:#58a6ff">Accuracy:</b>      {res['acc_test']:.4f} &nbsp;|&nbsp;
        <b style="color:#58a6ff">F1-Score:</b>       {res['f1_test']:.4f}
    </div>
    """, unsafe_allow_html=True)


# ── 4. BACKTEST DETALHADO ─────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title">💼 Backtest Realista</div>', unsafe_allow_html=True)

b1, b2, b3, b4, b5 = st.columns(5)
bt_ret_cor = "positive" if bt['ret_total'] >= 0 else "negative"
for col, label, val, cor in [
    (b1, "Capital Final",   f"${bt['capital_final']:,.0f}", "neutral"),
    (b2, "Retorno Total",   fmt_pct(bt['ret_total']),       bt_ret_cor),
    (b3, "Nº de Trades",    f"{bt['n_trades']:,}",          "neutral"),
    (b4, "Win Rate",        f"{bt['win_rate']:.1%}",        "positive" if bt['win_rate'] > 0.5 else "warning"),
    (b5, "Max Drawdown BT", fmt_pct(bt['max_dd']),          "negative"),
]:
    col.markdown(f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {cor}">{val}</div>
    </div>""", unsafe_allow_html=True)

# Equity curve do backtest realista
eq_bt = bt['equity_curve']
if len(eq_bt) > 1:
    st.markdown("<br>", unsafe_allow_html=True)
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(
        y=list(eq_bt), mode='lines',
        name='Capital (Backtest Realista)',
        line=dict(color='#bc8cff', width=2),
        fill='tozeroy',
        fillcolor='rgba(188,140,255,0.05)',
    ))
    fig_bt.update_layout(
        title=f'Backtest Realista | Threshold: {threshold:.0%} | Capital Inicial: ${capital:,.0f}',
        paper_bgcolor='#0b0e14', plot_bgcolor='#131720',
        font=dict(color='#f0f6fc', family='Space Mono', size=11),
        xaxis=dict(gridcolor='#252d3d', title='Barra'),
        yaxis=dict(gridcolor='#252d3d', tickprefix='$', tickformat=',.0f'),
        height=350, margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_bt, use_container_width=True)


# ── 5. DADOS BRUTOS (OHLCV) ───────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📋 Visualizar Dados Brutos (OHLCV)"):
    df_raw = res['df_raw']
    if 'Close' in df_raw.columns:
        fig_ohlc = go.Figure(data=[go.Candlestick(
            x=df_raw.index[-250:],
            open=df_raw['Open'].iloc[-250:],
            high=df_raw['High'].iloc[-250:],
            low=df_raw['Low'].iloc[-250:],
            close=df_raw['Close'].iloc[-250:],
            name=TICKER,
            increasing_line_color='#3fb950',
            decreasing_line_color='#f85149',
        )])
        fig_ohlc.update_layout(
            title=f'{TICKER} — Candlestick (últimas 250 barras)',
            paper_bgcolor='#0b0e14', plot_bgcolor='#131720',
            font=dict(color='#f0f6fc', family='Space Mono', size=11),
            xaxis=dict(gridcolor='#252d3d', rangeslider_visible=False),
            yaxis=dict(gridcolor='#252d3d'),
            height=400, margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_ohlc, use_container_width=True)
    st.dataframe(df_raw.tail(50).sort_index(ascending=False),
                 use_container_width=True, height=300)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#555; font-size:11px; font-family:Space Mono; border-top:1px solid #252d3d; padding-top:16px;'>
    ⚠ AVISO LEGAL: Este sistema é para fins educacionais e de pesquisa quantitativa.
    Resultados passados não garantem resultados futuros.
    NÃO constitui recomendação de investimento.
</div>
""", unsafe_allow_html=True)
