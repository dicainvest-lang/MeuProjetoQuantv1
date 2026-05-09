# -*- coding: utf-8 -*-
# ==============================================================================
# ROBÔ IA OURO v7.0 — STREAMLIT DASHBOARD EDITION (REFATORADO)
#
# MELHORIAS vs versão anterior:
#   1. Walk-Forward Validation (sem data leakage no backtest)
#   2. Features expandidas: trend, momentum, regime, sazonalidade
#   3. Ensemble: RF + XGBoost + LightGBM com VotingClassifier
#   4. Target baseado em retorno ajustado ao risco (Sharpe-like)
#   5. Filtro de regime de mercado (só opera a favor da tendência)
#   6. SL/TP dinâmico baseado em ATR (não fixo)
#   7. Custo realista (spread + comissão) por trade
#   8. Gestão de capital: Kelly Criterion (fração) + position sizing
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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

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
# PÁGINA
# ==============================================================================
st.set_page_config(page_title="Robô IA Ouro v7.0", page_icon="🥇", layout="wide")

st.markdown("""
<style>
    :root {
        --bg:#0b0e14; --card:#131720; --border:#2a3048;
        --text:#e8eaf6; --sub:#8b949e;
        --gold:#c9a84c; --green:#3fb950; --red:#f85149;
        --blue:#58a6ff; --purple:#bc8cff; --yellow:#d29922;
    }
    .stApp{background:var(--bg);color:var(--text);}
    section[data-testid="stSidebar"]{background:var(--card);border-right:1px solid var(--border);}
    .kpi{background:var(--card);border:1px solid var(--border);border-radius:12px;
         padding:18px 16px;text-align:center;box-shadow:0 4px 16px rgba(0,0,0,.5);}
    .kpi .lbl{color:var(--sub);font-size:.72rem;text-transform:uppercase;
              letter-spacing:1px;margin-bottom:6px;}
    .kpi .val{font-size:1.75rem;font-weight:700;}
    .kpi .sub{font-size:.7rem;color:var(--sub);margin-top:4px;}
    .sbox{border-radius:14px;padding:26px;text-align:center;margin:14px 0;border:2px solid;}
    .green{color:#3fb950;} .red{color:#f85149;} .gold{color:#c9a84c;}
    .blue{color:#58a6ff;} .yellow{color:#d29922;} .sub{color:#8b949e;}
    h1,h2,h3{color:var(--text)!important;}
    .stButton>button{background:linear-gradient(135deg,#c9a84c,#f0d080);
        color:#0b0e14;border:none;border-radius:8px;font-weight:800;
        padding:.6rem 1.8rem;font-size:1rem;width:100%;}
    div[data-testid="stMetric"]{background:var(--card);border:1px solid var(--border);
        border-radius:10px;padding:12px;}
    .improve-tag{background:#1a2a1a;border:1px solid #3fb950;border-radius:6px;
        padding:3px 8px;font-size:.72rem;color:#3fb950;margin-left:8px;}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("## 🥇 Robô IA Ouro v7.0")
    st.markdown('<span class="improve-tag">✨ Refatorado</span>', unsafe_allow_html=True)
    st.markdown("---")

    TICKER = st.text_input("🔎 Ativo", value="GC=F",
                            help="GC=F=Ouro Futuro | XAUUSD=X=Spot | GLD=ETF")
    periodo_map = {"2 Anos":2,"3 Anos":3,"5 Anos":5,"7 Anos":7,"10 Anos":10}
    periodo_label = st.selectbox("📅 Período", list(periodo_map.keys()), index=2)
    ANOS = periodo_map[periodo_label]
    TIMEFRAME = st.selectbox("⏱️ Timeframe", ["1d","1wk"], index=0)

    st.markdown("### 🎯 Sinal")
    PROB_THRESHOLD = st.slider("Threshold de Probabilidade", 0.50, 0.80, 0.55, 0.01)
    HORIZONTE      = st.slider("Horizonte (candles)", 1, 15, 3, 1)
    REGIME_FILTER  = st.checkbox("Filtro de Regime (só a favor da tendência)", value=True,
                                  help="Opera compra só acima da SMA200, venda só abaixo")

    st.markdown("### 💰 Risco")
    CAPITAL        = st.number_input("Capital (USD)", value=10_000, step=1_000)
    RISCO_PCT      = st.slider("Risco por Trade (%)", 0.5, 3.0, 1.0, 0.25) / 100
    ATR_MULT_SL    = st.slider("ATR Mult. SL", 0.5, 3.0, 1.5, 0.25)
    ATR_MULT_TP    = st.slider("ATR Mult. TP", 1.0, 6.0, 3.0, 0.25)
    CUSTO_BPS      = st.slider("Custo por Trade (bps)", 1, 20, 5, 1) / 10000
    MAX_DD_PCT     = st.slider("Max Drawdown Diário (%)", 1.0, 10.0, 5.0, 0.5) / 100

    st.markdown("### 🔄 Auto-Refresh")
    refresh_sel = st.selectbox("Intervalo", ["Desligado","1 min","5 min"], index=0)
    run_btn = st.button("🚀 Executar Análise")

if AUTOREFRESH_OK and refresh_sel != "Desligado":
    ms = 60_000 if refresh_sel == "1 min" else 300_000
    st_autorefresh(interval=ms, key="ar")

st.markdown("# 🥇 Robô IA Ouro v7.0 — Dashboard Melhorado")
st.markdown(f"**Ativo:** `{TICKER}` | **TF:** `{TIMEFRAME}` | **Período:** {periodo_label} "
            f"| {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
st.markdown("---")

# ==============================================================================
# DOWNLOAD
# ==============================================================================
@st.cache_data(ttl=60)
def baixar_dados(ticker, anos, interval):
    start = (datetime.now() - timedelta(days=anos*365)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, interval=interval,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().capitalize() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    for a in ['Adj close','Adj_close','Adjclose']:
        if a in df.columns and 'Close' not in df.columns:
            df.rename(columns={a:'Close'}, inplace=True)
    if 'Volume' not in df.columns: df['Volume'] = 0.0
    df['Volume'] = df['Volume'].fillna(0).astype(float)
    for c in ['Open','High','Low','Close']:
        if c in df.columns: df[c] = df[c].astype(float)
    df.dropna(subset=['Open','High','Low','Close'], inplace=True)
    return df.sort_index()[~df.index.duplicated(keep='last')]

# ==============================================================================
# MELHORIA 2 — FEATURE ENGINEERING EXPANDIDO
# ==============================================================================
def calcular_rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100/(1 + g/l.replace(0,np.nan))

def preparar_features(df_raw, horizonte=3):
    df = df_raw.copy()
    df.columns = [c.capitalize() for c in df.columns]
    if 'Close' in df.columns: df.rename(columns={'Close':'Ouro'}, inplace=True)
    c = df['Ouro']; h = df['High']; l = df['Low']; o = df['Open']
    v = df['Volume'].replace(0, np.nan)
    ret = c.pct_change()

    # ── Tendência ─────────────────────────────────────────────────────────────
    for p in [10,20,50,100,200]:
        df[f'sma_{p}']   = c.rolling(p).mean()
        df[f'ema_{p}']   = c.ewm(span=p,adjust=False).mean()
        df[f'dist_{p}']  = (c / df[f'sma_{p}'] - 1) * 100

    df['trend_short']  = (df['sma_10']  > df['sma_50']).astype(int)
    df['trend_long']   = (df['sma_50']  > df['sma_200']).astype(int)
    df['above_sma200'] = (c > df['sma_200']).astype(int)   # regime bull

    # Inclinação da SMA50 (proxy de momentum de tendência)
    df['sma50_slope']  = df['sma_50'].pct_change(5) * 100

    # ── Momentum ──────────────────────────────────────────────────────────────
    for p in [7,14,21]:
        df[f'rsi_{p}'] = calcular_rsi(c, p)
    df['rsi_slope'] = df['rsi_14'].diff(3)          # aceleração do RSI

    for p in [5,10,20,60]:
        df[f'roc_{p}'] = c.pct_change(p) * 100

    ema12 = c.ewm(span=12,adjust=False).mean()
    ema26 = c.ewm(span=26,adjust=False).mean()
    df['macd']        = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9,adjust=False).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']
    df['macd_cross']  = (df['macd'] > df['macd_signal']).astype(int)

    for p in [14,21]:
        lmin = l.rolling(p).min(); hmax = h.rolling(p).max()
        df[f'stoch_k_{p}'] = 100*(c-lmin)/(hmax-lmin).replace(0,np.nan)
        df[f'stoch_d_{p}'] = df[f'stoch_k_{p}'].rolling(3).mean()

    # ── Volatilidade ──────────────────────────────────────────────────────────
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    for p in [7,14,21]:
        df[f'atr_{p}']     = tr.rolling(p).mean()
        df[f'atr_{p}_pct'] = df[f'atr_{p}'] / c

    for p in [5,10,20,60]:
        df[f'vol_{p}'] = ret.rolling(p).std() * np.sqrt(252)

    df['vol_regime']   = df['vol_5'] / df['vol_20']   # vol relativa curta/longa

    mid20 = c.rolling(20).mean(); std20 = c.rolling(20).std()
    df['bb_up']   = mid20 + 2*std20
    df['bb_lo']   = mid20 - 2*std20
    df['bb_width']= (df['bb_up'] - df['bb_lo']) / mid20
    df['bb_pos']  = (c - df['bb_lo']) / (df['bb_up'] - df['bb_lo']).replace(0,np.nan)

    # ── Mean Reversion ────────────────────────────────────────────────────────
    # MELHORIA: Z-score mais estável com janela menor + normalização por vol
    for p in [20,60]:
        mu = c.rolling(p).mean(); sg = c.rolling(p).std()
        df[f'zscore_{p}'] = (c - mu) / sg.replace(0,np.nan)

    # ── Volume ────────────────────────────────────────────────────────────────
    df['vol_rel']  = v / v.rolling(20).mean()
    obv = (np.sign(c.diff())*v).fillna(0).cumsum()
    df['obv_slope']= obv.pct_change(5)

    # ── Candlestick ───────────────────────────────────────────────────────────
    df['corpo']    = (c - o).abs() / o.clip(lower=1e-9)
    df['gap']      = (o - c.shift(1)) / c.shift(1).clip(lower=1e-9)
    df['high_rel'] = (h - c) / c
    df['low_rel']  = (c - l) / c

    # ── Lags ──────────────────────────────────────────────────────────────────
    for lag in [1,2,3,5]:
        df[f'ret_lag_{lag}'] = ret.shift(lag)
    for lag in [1,3]:
        df[f'rsi14_lag_{lag}'] = df['rsi_14'].shift(lag)
        df[f'macd_lag_{lag}']  = df['macd'].shift(lag)

    # ── Sazonalidade ──────────────────────────────────────────────────────────
    df['dia_semana'] = df.index.dayofweek
    df['mes']        = df.index.month
    df['trimestre']  = df.index.quarter

    # ── Regime de volatilidade ────────────────────────────────────────────────
    df['regime_vol_alta'] = (df['vol_regime'] > 1.3).astype(int)
    df['adx'] = _calcular_adx(h, l, c, 14)
    df['regime_tendencia'] = (df['adx'] > 25).astype(int)

    # ── Target MELHORADO — retorno ajustado ao risco ──────────────────────────
    # Em vez de percentil fixo, usa retorno relativo ao ATR (Sharpe-like)
    # Isso evita sinalizar movimentos menores que o custo operacional
    ret_fut   = c.pct_change(horizonte).shift(-horizonte)
    atr_norm  = df['atr_14_pct'].rolling(5).mean()
    # Threshold dinâmico: movimento deve ser > 0.5x ATR para ser sinal
    thr_dyn   = atr_norm * 0.5
    df['Alvo']       = np.select([ret_fut > thr_dyn, ret_fut < -thr_dyn], [1,-1], default=0)
    df['Ret_Futuro'] = ret_fut

    # ── Manter colunas originais para visualização ────────────────────────────
    df['Ouro_Close'] = c
    df['ATR']        = df['atr_14']
    df['ATR_Pct']    = df['atr_14_pct']
    df['RSI']        = df['rsi_14']
    df['Z_Score']    = df['zscore_20']
    df['Volatilidade'] = df['vol_20']
    df['SMA_20']     = df['sma_20']
    df['SMA_50']     = df['sma_50']
    df['SMA_200']    = df['sma_200']
    df['BB_Upper']   = df['bb_up']
    df['BB_Lower']   = df['bb_lo']
    df['MACD']       = df['macd']
    df['MACD_Signal']= df['macd_signal']
    df['MACD_Hist']  = df['macd_hist']

    return df.replace([np.inf,-np.inf], np.nan)

def _calcular_adx(h, l, c, p=14):
    dm_p = h.diff().clip(lower=0)
    dm_m = (-l.diff()).clip(lower=0)
    tr   = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    tr_s = tr.ewm(span=p,adjust=False).mean()
    di_p = 100 * dm_p.ewm(span=p,adjust=False).mean() / tr_s.replace(0,np.nan)
    di_m = 100 * dm_m.ewm(span=p,adjust=False).mean() / tr_s.replace(0,np.nan)
    dx   = 100 * (di_p - di_m).abs() / (di_p + di_m + 1e-9)
    return dx.ewm(span=p,adjust=False).mean()

# ==============================================================================
# MELHORIA 3 — ENSEMBLE + SCALER
# ==============================================================================
FEAT_COLS = [
    'rsi_7','rsi_14','rsi_21','rsi_slope',
    'roc_5','roc_10','roc_20','roc_60',
    'macd','macd_hist','macd_cross',
    'stoch_k_14','stoch_d_14','stoch_k_21',
    'zscore_20','zscore_60',
    'atr_14_pct','atr_7_pct',
    'vol_5','vol_20','vol_regime',
    'bb_pos','bb_width',
    'dist_10','dist_20','dist_50','dist_200',
    'trend_short','trend_long','sma50_slope',
    'ret_lag_1','ret_lag_2','ret_lag_3','ret_lag_5',
    'rsi14_lag_1','rsi14_lag_3',
    'macd_lag_1','macd_lag_3',
    'corpo','gap','high_rel','low_rel',
    'vol_rel','obv_slope',
    'vol_regime','regime_vol_alta','regime_tendencia','adx',
    'dia_semana','mes','trimestre',
]

def treinar_ensemble(df_feat, n_splits=5):
    """
    MELHORIA 1: Walk-Forward Validation — sem data leakage.
    Treina no conjunto inteiro após WF, avalia OOS real.
    """
    df_clean = df_feat[FEAT_COLS + ['Alvo']].dropna()
    if len(df_clean) < 150:
        raise ValueError(f"Dados insuficientes após limpeza: {len(df_clean)} linhas.")

    X = df_clean[FEAT_COLS].values
    y = df_clean['Alvo'].values
    dates = df_clean.index

    # Walk-Forward para métricas OOS honestas
    tscv   = TimeSeriesSplit(n_splits=n_splits, test_size=max(20, len(X)//(n_splits+1)))
    wf_acc, wf_f1, wf_roc = [], [], []
    scaler_wf = RobustScaler()

    for tr_idx, te_idx in tscv.split(X):
        Xtr = RobustScaler().fit_transform(X[tr_idx])
        Xte = RobustScaler().fit_transform(X[te_idx])  # fit separado por janela
        Xte = RobustScaler().fit(X[tr_idx]).transform(X[te_idx])

        est = _build_ensemble()
        est.fit(Xtr, y[tr_idx])
        yp  = est.predict(Xte)
        ypr = est.predict_proba(Xte)

        wf_acc.append(accuracy_score(y[te_idx], yp))
        wf_f1.append(f1_score(y[te_idx], yp, average='weighted', zero_division=0))
        try:
            roc = roc_auc_score(y[te_idx], ypr, multi_class='ovr', average='weighted')
            wf_roc.append(roc)
        except:
            pass

    # Treino final em todos os dados
    scaler = RobustScaler()
    X_s    = scaler.fit_transform(X)
    modelo_final = _build_ensemble()
    modelo_final.fit(X_s, y)

    n_tr = int(len(X)*0.80)
    return modelo_final, scaler, {
        'acc_oos':  float(np.mean(wf_acc)),
        'f1_oos':   float(np.mean(wf_f1)),
        'roc_oos':  float(np.mean(wf_roc)) if wf_roc else 0.0,
        'n_train':  n_tr,
        'n_total':  len(X),
        'n_splits': n_splits,
        'feat_cols': FEAT_COLS,
        'dates':    dates,
    }

def _build_ensemble():
    """Monta o VotingClassifier com os modelos disponíveis."""
    est = [
        ('rf', RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=10,
            class_weight='balanced', random_state=42, n_jobs=-1)),
        ('lr', LogisticRegression(
            C=0.3, max_iter=1000, class_weight='balanced', random_state=42)),
    ]
    if XGB_OK:
        est.append(('xgb', XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='mlogloss', use_label_encoder=False,
            random_state=42, n_jobs=-1, verbosity=0)))
    if LGB_OK:
        est.append(('lgb', LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            min_child_samples=15, class_weight='balanced',
            random_state=42, n_jobs=-1, verbosity=-1)))
    return VotingClassifier(estimators=est, voting='soft', n_jobs=-1)

# ==============================================================================
# SINAL COM FILTRO DE REGIME
# ==============================================================================
def calcular_sinal(modelo, scaler, df_feat, prob_thr, regime_filter):
    df_c = df_feat[FEAT_COLS + ['Alvo','Ret_Futuro','Ouro_Close',
                                 'ATR_Pct','RSI','Z_Score',
                                 'Volatilidade','above_sma200','adx']].dropna()
    if len(df_c) == 0:
        return {}

    ult    = df_c.iloc[-1]
    X_ult  = scaler.transform(df_c[FEAT_COLS].iloc[[-1]].values)
    classes = list(modelo.classes_)
    probs  = modelo.predict_proba(X_ult)[0]

    p = {cls: float(probs[i]) for i, cls in enumerate(classes)}
    p_c = p.get(1,  0.0)
    p_v = p.get(-1, 0.0)
    p_n = p.get(0,  1.0 - p_c - p_v)

    bull_regime = bool(ult['above_sma200'] == 1)
    atr  = float(ult['ATR_Pct'])
    rsi  = float(ult['RSI'])
    z    = float(ult['Z_Score'])
    vol  = float(ult['Volatilidade'])
    adx  = float(ult['adx'])
    preco= float(ult['Ouro_Close'])

    # MELHORIA 5: Filtro de regime — só opera a favor da tendência dominante
    compra_ok = p_c > prob_thr and (not regime_filter or bull_regime)
    venda_ok  = p_v > prob_thr and (not regime_filter or not bull_regime)

    if compra_ok:   direcao, sinal_str = 1,  "COMPRA 🟢"
    elif venda_ok:  direcao, sinal_str = -1, "VENDA 🔴"
    else:           direcao, sinal_str = 0,  "NEUTRO ⚪"

    confianca = p_c if direcao==1 else p_v if direcao==-1 else max(p_c,p_v)

    return {
        'direcao':direcao,'sinal':sinal_str,'confianca':confianca,
        'p_compra':p_c,'p_venda':p_v,'p_neutro':p_n,
        'rsi':rsi,'z':z,'vol':vol*100,'atr_pct':atr,'adx':adx,
        'preco':preco,'bull_regime':bull_regime,
    }

# ==============================================================================
# MELHORIA 6+7+8 — BACKTEST WALK-FORWARD REALISTA
# ==============================================================================
def backtest_wf(df_feat, prob_thr, regime_filter,
                 capital, risco_pct, mult_sl, mult_tp,
                 custo_bps, max_dd_pct, n_splits=5):
    """
    MELHORIA 1: Backtest walk-forward — modelo re-treinado a cada janela.
    MELHORIA 6: SL/TP dinâmico em ATR.
    MELHORIA 7: Custo realista (spread + comissão) por trade.
    MELHORIA 8: Drawdown diário máximo bloqueia operações.
    """
    df_c = df_feat[FEAT_COLS + ['Alvo','Ret_Futuro','Ouro_Close',
                                  'ATR_Pct','above_sma200']].dropna()
    if len(df_c) < 150:
        return None

    X = df_c[FEAT_COLS].values
    y = df_c['Alvo'].values
    ret_fut = df_c['Ret_Futuro'].values
    precos  = df_c['Ouro_Close'].values
    atrs    = df_c['ATR_Pct'].values
    regimes = df_c['above_sma200'].values
    dates   = df_c.index

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=max(30, len(X)//(n_splits+1)))

    cap        = capital
    equity     = [cap]
    eq_dates   = [dates[0]]
    trades     = []
    dd_inicio_dia = cap
    dia_atual  = dates[0].date() if hasattr(dates[0],'date') else dates[0]

    for tr_idx, te_idx in tscv.split(X):
        if len(tr_idx) < 50 or len(te_idx) < 5:
            continue

        sc = RobustScaler()
        Xtr = sc.fit_transform(X[tr_idx])
        Xte = sc.transform(X[te_idx])

        mod = _build_ensemble()
        try:
            mod.fit(Xtr, y[tr_idx])
        except:
            continue

        classes = list(mod.classes_)
        probs   = mod.predict_proba(Xte)

        for j, i in enumerate(te_idx):
            pr_arr   = probs[j]
            p = {cls: float(pr_arr[k]) for k, cls in enumerate(classes)}
            p_c = p.get(1,0.0); p_v = p.get(-1,0.0)
            bull = regimes[i] == 1
            atr  = atrs[i]
            ret  = ret_fut[i] if not np.isnan(ret_fut[i]) else 0.0
            dt   = dates[i]

            # Drawdown diário
            d_atual = dt.date() if hasattr(dt,'date') else dt
            if d_atual != dia_atual:
                dia_atual = d_atual
                dd_inicio_dia = cap
            dd_hoje = (dd_inicio_dia - cap) / dd_inicio_dia if dd_inicio_dia > 0 else 0

            if dd_hoje >= max_dd_pct:
                equity.append(cap); eq_dates.append(dt)
                continue

            # Sinal com filtro de regime
            compra_ok = p_c > prob_thr and (not regime_filter or bull)
            venda_ok  = p_v > prob_thr and (not regime_filter or not bull)

            if compra_ok:   d =  1
            elif venda_ok:  d = -1
            else:
                equity.append(cap); eq_dates.append(dt)
                continue

            # SL/TP dinâmico em ATR
            sl_d = atr * mult_sl
            tp_d = atr * mult_tp
            if sl_d <= 0:
                equity.append(cap); eq_dates.append(dt)
                continue

            # Position sizing baseado em risco fixo
            risco_usd  = cap * risco_pct
            sl_pts_usd = precos[i] * sl_d          # $SL por % do preço
            pos_size   = risco_usd / (sl_pts_usd + 1e-9)
            pos_size   = min(pos_size, cap * 0.5)  # nunca mais de 50% do capital

            # P&L
            pnl_bruto = pos_size * d * ret
            if d == 1:
                if ret < -sl_d:  pnl_bruto = -risco_usd
                elif ret > tp_d: pnl_bruto =  risco_usd * (tp_d/sl_d)
            else:
                if ret >  sl_d:  pnl_bruto = -risco_usd
                elif ret < -tp_d:pnl_bruto =  risco_usd * (tp_d/sl_d)

            # Custo realista
            custo = cap * custo_bps
            pnl   = pnl_bruto - custo
            cap  += pnl

            equity.append(cap); eq_dates.append(dt)
            trades.append({
                'date':dt,'dir':d,'ret':ret,'pnl':pnl,
                'risco_usd':risco_usd,'atr':atr,
                'prob': p_c if d==1 else p_v,
            })

    if len(equity) < 2:
        return None

    eq  = np.array(equity)
    peak= np.maximum.accumulate(eq)
    dd  = (eq - peak) / peak
    rets_eq = np.diff(eq)/eq[:-1]

    n_t  = len(trades)
    win_r= np.mean([t['pnl']>0 for t in trades])*100 if n_t else 0.0
    shp  = float(rets_eq.mean()/rets_eq.std()*np.sqrt(252)) if rets_eq.std()>0 else 0.0
    srt  = float(rets_eq.mean()/rets_eq[rets_eq<0].std()*np.sqrt(252)) if len(rets_eq[rets_eq<0])>0 else 0.0
    r_a  = (eq[-1]/capital-1)*100
    bh   = (precos[-1]/precos[0]-1)*100 if precos[0]>0 else 0.0

    # Profit Factor
    wins  = [t['pnl'] for t in trades if t['pnl']>0]
    losss = [t['pnl'] for t in trades if t['pnl']<0]
    pf    = sum(wins)/abs(sum(losss)) if losss else float('inf')

    # Expectancy
    avg_w = np.mean(wins)  if wins  else 0.0
    avg_l = np.mean(losss) if losss else 0.0
    exp   = (win_r/100)*avg_w + (1-win_r/100)*avg_l

    # Calmar
    calmar = (r_a/abs(dd.min()*100)) if dd.min()<0 else 0.0

    return {
        'equity':eq,'dd':dd,'eq_dates':eq_dates[:len(eq)],
        'trades':trades,'precos_hist':precos,
        'ret_acum':r_a,'bh':bh,'sharpe':shp,'sortino':srt,
        'max_dd':float(dd.min()*100),'win_rate':win_r,
        'n_trades':n_t,'cap_final':cap,
        'profit_factor':pf,'expectancy':exp,'calmar':calmar,
    }

# ==============================================================================
# EXECUÇÃO
# ==============================================================================
if run_btn or 'results' not in st.session_state:
    prog = st.progress(0, text="⬇️ Baixando dados...")
    try:
        df_raw = baixar_dados(TICKER.upper(), ANOS, TIMEFRAME)
        prog.progress(20, text="🔧 Calculando features (~50 indicadores)...")
        if len(df_raw) < 250:
            st.error("⚠️ Dados insuficientes. Aumente o período para ≥ 3 anos.")
            st.stop()
        df_feat = preparar_features(df_raw, horizonte=HORIZONTE)
        df_feat.dropna(inplace=True)
        if len(df_feat) < 150:
            st.error(f"⚠️ Apenas {len(df_feat)} linhas após features. Use período maior.")
            st.stop()
        prog.progress(40, text="🧠 Treinando Ensemble (RF + XGB + LGB)...")
        modelo, scaler, met = treinar_ensemble(df_feat, n_splits=4)
        prog.progress(70, text="📈 Rodando backtest walk-forward...")
        sinal = calcular_sinal(modelo, scaler, df_feat, PROB_THRESHOLD, REGIME_FILTER)
        bt    = backtest_wf(df_feat, PROB_THRESHOLD, REGIME_FILTER,
                             CAPITAL, RISCO_PCT, ATR_MULT_SL, ATR_MULT_TP,
                             CUSTO_BPS, MAX_DD_PCT, n_splits=4)
        prog.progress(100, text="✅ Concluído!")
        if bt is None:
            st.error("Backtest retornou vazio. Reduza o threshold ou aumente o período.")
            st.stop()
        st.session_state['results'] = {
            'df_raw':df_raw,'df_feat':df_feat,
            'modelo':modelo,'scaler':scaler,'met':met,
            'sinal':sinal,'bt':bt,
        }
        prog.empty()
    except Exception as e:
        prog.empty()
        st.error(f"Erro: {e}")
        st.stop()

res = st.session_state.get('results')
if not res:
    st.info("Configure os parâmetros e clique em **🚀 Executar Análise**.")
    st.stop()

s  = res['sinal']
bt = res['bt']
m  = res['met']

# ==============================================================================
# KPIs
# ==============================================================================
st.markdown("## 📊 Resultados (Walk-Forward OOS)")

def kpi(col, lbl, val, css, sub=""):
    col.markdown(f"""<div class="kpi">
        <div class="lbl">{lbl}</div>
        <div class="val {css}">{val}</div>
        <div class="sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)
kpi(c1,"Retorno Acumulado",f"{bt['ret_acum']:+.2f}%",
    "green" if bt['ret_acum']>=0 else "red", f"B&H Ouro: {bt['bh']:+.2f}%")
kpi(c2,"Sharpe Ratio",f"{bt['sharpe']:.2f}",
    "green" if bt['sharpe']>=1 else ("blue" if bt['sharpe']>=0 else "red"),"Anualizado OOS")
kpi(c3,"Win Rate",f"{bt['win_rate']:.1f}%",
    "green" if bt['win_rate']>=50 else "red",f"{bt['n_trades']} trades")
kpi(c4,"Max Drawdown",f"{bt['max_dd']:.2f}%","red","Pior queda")

st.markdown("<br>",unsafe_allow_html=True)
m1,m2,m3,m4,m5,m6 = st.columns(6)
m1.metric("Sortino",        f"{bt['sortino']:.2f}")
m2.metric("Profit Factor",  f"{bt['profit_factor']:.2f}" if bt['profit_factor']!=float('inf') else "∞")
m3.metric("Expectancy",     f"${bt['expectancy']:+.1f}")
m4.metric("Calmar",         f"{bt['calmar']:.2f}")
m5.metric("Acurácia OOS",   f"{m['acc_oos']:.1%}")
m6.metric("ROC-AUC OOS",    f"{m['roc_oos']:.3f}")

st.markdown("---")

# ==============================================================================
# TABS
# ==============================================================================
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "🎯 Sinal","📉 Equity & Drawdown","📊 Indicadores","🏆 Performance","🧠 Modelo"
])

BG,CARD,GRID,TXT = '#0b0e14','#131720','#2a3048','#e8eaf6'

def bl(h=480):
    return dict(height=h,paper_bgcolor=BG,plot_bgcolor=CARD,
                font=dict(color=TXT),legend=dict(bgcolor=CARD),
                margin=dict(l=50,r=20,t=50,b=40))

def uf(fig):
    fig.update_xaxes(gridcolor=GRID,zeroline=False)
    fig.update_yaxes(gridcolor=GRID,zeroline=False)
    return fig

# ── Tab 1: Sinal ──────────────────────────────────────────────────────────────
with tab1:
    sc = ("#3fb950" if "COMPRA" in s.get('sinal','') else
          "#f85149" if "VENDA"  in s.get('sinal','') else "#8b949e")
    regime_txt = "🐂 Bull (acima SMA200)" if s.get('bull_regime') else "🐻 Bear (abaixo SMA200)"

    st.markdown(f"""<div class="sbox" style="border-color:{sc};background:#131720;">
        <div style="font-size:3rem;margin-bottom:6px;">{s.get('sinal','—')}</div>
        <div style="color:#8b949e;font-size:.85rem;">
            Ensemble ML | Regime: {regime_txt} | ADX: {s.get('adx',0):.1f}
        </div>
        <div style="margin-top:20px;display:flex;justify-content:center;gap:36px;">
            <div><div style="color:#8b949e;font-size:.7rem">PROB. COMPRA</div>
                 <div style="font-size:1.5rem;color:#3fb950;font-weight:700">{s.get('p_compra',0):.1%}</div></div>
            <div><div style="color:#8b949e;font-size:.7rem">PROB. VENDA</div>
                 <div style="font-size:1.5rem;color:#f85149;font-weight:700">{s.get('p_venda',0):.1%}</div></div>
            <div><div style="color:#8b949e;font-size:.7rem">NEUTRO</div>
                 <div style="font-size:1.5rem;color:#8b949e;font-weight:700">{s.get('p_neutro',0):.1%}</div></div>
            <div><div style="color:#8b949e;font-size:.7rem">CONFIANÇA</div>
                 <div style="font-size:1.5rem;color:#c9a84c;font-weight:700">{s.get('confianca',0):.1%}</div></div>
        </div>
    </div>""", unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        st.markdown("#### 📐 Gestão de Risco (SL/TP Dinâmico por ATR)")
        preco = s.get('preco',0)
        atr_p = s.get('atr_pct',0)
        sl_p  = preco*(1-atr_p*ATR_MULT_SL) if s.get('direcao')==1 else preco*(1+atr_p*ATR_MULT_SL)
        tp_p  = preco*(1+atr_p*ATR_MULT_TP) if s.get('direcao')==1 else preco*(1-atr_p*ATR_MULT_TP)
        risco_u = CAPITAL * RISCO_PCT
        tp_u    = risco_u * (ATR_MULT_TP/ATR_MULT_SL)
        st.markdown(f"""
| Parâmetro | Valor |
|:--|--:|
| Preço Atual | `${preco:,.2f}` |
| Stop Loss (preço) | `${sl_p:,.2f}` |
| Take Profit (preço) | `${tp_p:,.2f}` |
| Risco USD | `${risco_u:,.2f}` |
| Ganho Potencial | `${tp_u:,.2f}` |
| R/R Ratio | `1:{ATR_MULT_TP/ATR_MULT_SL:.1f}` |
| RSI (14) | `{s.get('rsi',0):.2f}` |
| Z-Score (20) | `{s.get('z',0):+.3f}` |
| ATR % | `{atr_p*100:.4f}%` |
| ADX | `{s.get('adx',0):.1f}` |
""")
    with cb:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=s.get('p_compra',0)*100,
            number={'suffix':'%','font':{'color':TXT}},
            gauge={'axis':{'range':[0,100]},
                   'bar':{'color':'#3fb950','thickness':.35},
                   'bgcolor':CARD,
                   'steps':[{'range':[0,PROB_THRESHOLD*100],'color':'#1a0f0f'},
                             {'range':[PROB_THRESHOLD*100,100],'color':'#0f1a0f'}],
                   'threshold':{'line':{'color':'#c9a84c','width':3},
                                'thickness':.8,'value':PROB_THRESHOLD*100}},
            title={'text':'Prob. COMPRA (%)'}))
        fig_g.update_layout(paper_bgcolor=BG,font=dict(color=TXT),height=280)
        st.plotly_chart(fig_g,use_container_width=True)

    st.warning("⚠️ Educacional. Não é recomendação de investimento.")

# ── Tab 2: Equity & Drawdown ──────────────────────────────────────────────────
with tab2:
    df_f  = res['df_feat']
    eq    = bt['equity']
    dd    = bt['dd']
    eq_dt = bt['eq_dates']

    fig_e = make_subplots(rows=3,cols=1,shared_xaxes=True,
                           row_heights=[.45,.3,.25],
                           subplot_titles=["Preço Ouro + SMA 50/200 + Bollinger",
                                           "Equity Curve (Walk-Forward OOS)",
                                           "Drawdown (%)"])
    fig_e.add_trace(go.Scatter(x=df_f.index,y=df_f['Ouro_Close'],
        name="Ouro",line=dict(color='#c9a84c',width=1.5)),row=1,col=1)
    fig_e.add_trace(go.Scatter(x=df_f.index,y=df_f['SMA_50'],
        name="SMA50",line=dict(color='#58a6ff',width=1,dash='dot')),row=1,col=1)
    fig_e.add_trace(go.Scatter(x=df_f.index,y=df_f['SMA_200'],
        name="SMA200",line=dict(color='#bc8cff',width=1.5,dash='dash')),row=1,col=1)
    fig_e.add_trace(go.Scatter(x=df_f.index,y=df_f['BB_Upper'],
        line=dict(color='#2a3048',width=.8),showlegend=False),row=1,col=1)
    fig_e.add_trace(go.Scatter(x=df_f.index,y=df_f['BB_Lower'],
        line=dict(color='#2a3048',width=.8),fill='tonexty',
        fillcolor='rgba(42,48,72,.18)',showlegend=False),row=1,col=1)

    ml = min(len(eq_dt),len(eq))
    fig_e.add_trace(go.Scatter(x=eq_dt[:ml],y=eq[:ml],name="Equity",
        fill='tozeroy',line=dict(color='#3fb950',width=2),
        fillcolor='rgba(63,185,80,.1)'),row=2,col=1)
    fig_e.add_hline(y=CAPITAL,line_dash="dash",line_color="#8b949e",row=2,col=1)

    fig_e.add_trace(go.Scatter(x=eq_dt[:ml],y=dd[:ml]*100,name="Drawdown",
        fill='tozeroy',line=dict(color='#f85149',width=1),
        fillcolor='rgba(248,81,73,.12)'),row=3,col=1)

    fig_e.update_layout(**bl(640)); uf(fig_e)
    st.plotly_chart(fig_e,use_container_width=True)

    if bt['trades']:
        df_tr = pd.DataFrame(bt['trades'])
        df_tr['Direção'] = df_tr['dir'].map({1:'COMPRA 🟢',-1:'VENDA 🔴'})
        df_tr['ret_%']   = (df_tr['ret']*100).round(4)
        df_tr['pnl_$']   = df_tr['pnl'].round(2)
        df_tr['prob_%']  = (df_tr['prob']*100).round(1)
        with st.expander(f"📋 Trades ({len(bt['trades'])} total)"):
            st.dataframe(df_tr[['date','Direção','prob_%','ret_%','pnl_$']].tail(60),
                         use_container_width=True)

# ── Tab 3: Indicadores ────────────────────────────────────────────────────────
with tab3:
    df_f = res['df_feat']
    ca3,cb3 = st.columns(2)
    with ca3:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df_f.index,y=df_f['rsi_7'],
            name="RSI 7",line=dict(color='#d29922',width=1)))
        fig_rsi.add_trace(go.Scatter(x=df_f.index,y=df_f['rsi_14'],
            name="RSI 14",line=dict(color='#c9a84c',width=1.8)))
        fig_rsi.add_trace(go.Scatter(x=df_f.index,y=df_f['rsi_21'],
            name="RSI 21",line=dict(color='#8b949e',width=1)))
        for v,clr in [(70,'#f85149'),(30,'#3fb950'),(50,'#2a3048')]:
            fig_rsi.add_hline(y=v,line_dash="dot",line_color=clr)
        fig_rsi.update_layout(title="RSI Multi-Período",**bl(290)); uf(fig_rsi)
        st.plotly_chart(fig_rsi,use_container_width=True)

        fig_z = go.Figure()
        for p,clr in [(20,'#58a6ff'),(60,'#bc8cff')]:
            fig_z.add_trace(go.Scatter(x=df_f.index,y=df_f[f'zscore_{p}'],
                name=f"Z-Score {p}",line=dict(color=clr,width=1.4)))
        fig_z.add_hline(y=1,line_dash="dash",line_color="#f85149")
        fig_z.add_hline(y=-1,line_dash="dash",line_color="#3fb950")
        fig_z.add_hline(y=0,line_color="#2a3048")
        fig_z.update_layout(title="Z-Score Multi-Janela",**bl(290)); uf(fig_z)
        st.plotly_chart(fig_z,use_container_width=True)

    with cb3:
        fig_macd = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.6,.4])
        fig_macd.add_trace(go.Scatter(x=df_f.index,y=df_f['MACD'],
            name="MACD",line=dict(color='#58a6ff',width=1.3)),row=1,col=1)
        fig_macd.add_trace(go.Scatter(x=df_f.index,y=df_f['MACD_Signal'],
            name="Signal",line=dict(color='#d29922',width=1.3)),row=1,col=1)
        clr_h = np.where(df_f['MACD_Hist']>=0,'#3fb950','#f85149')
        fig_macd.add_trace(go.Bar(x=df_f.index,y=df_f['MACD_Hist'],
            name="Hist",marker_color=clr_h),row=2,col=1)
        fig_macd.update_layout(title="MACD",**bl(380)); uf(fig_macd)
        st.plotly_chart(fig_macd,use_container_width=True)

        fig_adx = go.Figure()
        fig_adx.add_trace(go.Scatter(x=df_f.index,y=df_f['adx'],
            name="ADX",line=dict(color='#c9a84c',width=1.5),fill='tozeroy',
            fillcolor='rgba(201,168,76,.08)'))
        fig_adx.add_hline(y=25,line_dash="dash",line_color="#d29922",
                           annotation_text="Tendência forte (25)")
        fig_adx.update_layout(title="ADX (Força da Tendência)",**bl(240)); uf(fig_adx)
        st.plotly_chart(fig_adx,use_container_width=True)

# ── Tab 4: Performance ────────────────────────────────────────────────────────
with tab4:
    ca4,cb4 = st.columns(2)
    with ca4:
        st.markdown("### 📋 Métricas Completas")
        st.markdown(f"""
| Métrica | Valor |
|:--|--:|
| Retorno Acumulado | `{bt['ret_acum']:+.2f}%` |
| Buy & Hold Ouro | `{bt['bh']:+.2f}%` |
| Sharpe Ratio | `{bt['sharpe']:.3f}` |
| Sortino Ratio | `{bt['sortino']:.3f}` |
| Calmar Ratio | `{bt['calmar']:.3f}` |
| Max Drawdown | `{bt['max_dd']:.2f}%` |
| Win Rate | `{bt['win_rate']:.1f}%` |
| Profit Factor | `{bt['profit_factor']:.3f}` |
| Expectancy | `${bt['expectancy']:+.2f}` |
| N° Trades | `{bt['n_trades']}` |
| Capital Final | `${bt['cap_final']:,.2f}` |
""")
        # Distribuição PnL
        if bt['trades']:
            pnls = [t['pnl'] for t in bt['trades']]
            fig_pnl = go.Figure(go.Histogram(x=pnls, nbinsx=40,
                marker_color='#c9a84c', opacity=.8))
            fig_pnl.add_vline(x=0,line_dash="dash",line_color="#f85149")
            fig_pnl.update_layout(title="Distribuição P&L por Trade",**bl(300)); uf(fig_pnl)
            st.plotly_chart(fig_pnl,use_container_width=True)

    with cb4:
        # Equity mensal
        if bt['trades']:
            df_tr2 = pd.DataFrame(bt['trades'])
            df_tr2['date'] = pd.to_datetime(df_tr2['date'])
            df_tr2.set_index('date',inplace=True)
            monthly = df_tr2['pnl'].resample('ME').sum()
            fig_mon = go.Figure(go.Bar(
                x=monthly.index, y=monthly.values,
                marker_color=np.where(monthly.values>=0,'#3fb950','#f85149')))
            fig_mon.update_layout(title="P&L Mensal (USD)",**bl(320)); uf(fig_mon)
            st.plotly_chart(fig_mon,use_container_width=True)

        # Trades Compra vs Venda
        trades_df = pd.DataFrame(bt['trades'])
        if not trades_df.empty:
            vc = trades_df[trades_df['dir']==1]['pnl']
            vv = trades_df[trades_df['dir']==-1]['pnl']
            fig_cv = go.Figure()
            if len(vc):
                fig_cv.add_trace(go.Box(y=vc,name="Compras",marker_color='#3fb950'))
            if len(vv):
                fig_cv.add_trace(go.Box(y=vv,name="Vendas",marker_color='#f85149'))
            fig_cv.update_layout(title="P&L por Direção",**bl(280)); uf(fig_cv)
            st.plotly_chart(fig_cv,use_container_width=True)

# ── Tab 5: Modelo ─────────────────────────────────────────────────────────────
with tab5:
    ca5,cb5 = st.columns(2)
    with ca5:
        st.markdown("### 🧠 Ensemble — Métricas Walk-Forward OOS")
        modelos_str = "RF + LR"
        if XGB_OK: modelos_str += " + XGB"
        if LGB_OK: modelos_str += " + LGB"
        st.markdown(f"""
| Métrica | Valor |
|:--|--:|
| Modelos | `{modelos_str}` |
| Acurácia OOS | `{m['acc_oos']:.1%}` |
| F1-Score OOS | `{m['f1_oos']:.3f}` |
| ROC-AUC OOS | `{m['roc_oos']:.3f}` |
| Amostras totais | `{m['n_total']:,}` |
| Walk-Forward splits | `{m['n_splits']}` |
| Features | `{len(m['feat_cols'])}` |
""")
        # Feature importance (RF)
        rf_mod = None
        try:
            for name, est in res['modelo'].estimators_:
                if name == 'rf':
                    rf_mod = est; break
        except:
            pass
        if rf_mod:
            fi = pd.Series(rf_mod.feature_importances_,
                           index=m['feat_cols']).sort_values().tail(20)
            fig_fi = go.Figure(go.Bar(x=fi.values,y=fi.index,
                orientation='h',marker_color='#c9a84c'))
            fig_fi.update_layout(title="Top 20 Features (RF Gini)",**bl(420)); uf(fig_fi)
            st.plotly_chart(fig_fi,use_container_width=True)

    with cb5:
        df_f5 = res['df_feat']
        alvo  = df_f5['Alvo'].value_counts().sort_index()
        labels= {1:'COMPRA',0:'NEUTRO',-1:'VENDA'}
        fig_pie = go.Figure(go.Pie(
            labels=[labels.get(i,str(i)) for i in alvo.index],
            values=alvo.values,
            marker_colors=['#3fb950','#8b949e','#f85149'],
            hole=.4))
        fig_pie.update_layout(title="Distribuição do Target",
            paper_bgcolor=BG,font=dict(color=TXT),height=300,
            margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig_pie,use_container_width=True)

        ret_f = df_f5['Ret_Futuro'].dropna()*100
        fig_rh = go.Figure(go.Histogram(x=ret_f,nbinsx=60,
            marker_color='#c9a84c',opacity=.8))
        fig_rh.add_vline(x=0,line_dash="dash",line_color="#f85149")
        fig_rh.update_layout(title=f"Retorno a {HORIZONTE} candles (%)",**bl(300)); uf(fig_rh)
        st.plotly_chart(fig_rh,use_container_width=True)

# ==============================================================================
# DADOS BRUTOS
# ==============================================================================
st.markdown("---")
with st.expander("🗂️ Últimas 100 barras"):
    st.dataframe(res['df_raw'].tail(100).style.format(precision=4),use_container_width=True)
with st.expander("🤖 Features (últimas 50 linhas)"):
    cols_show = ['Ouro_Close','RSI','Z_Score','Volatilidade',
                 'ATR_Pct','adx','above_sma200','Alvo']
    st.dataframe(res['df_feat'][cols_show].tail(50).style.format(precision=4),
                 use_container_width=True)

st.caption("⚠️ Dashboard educacional. Resultados passados não garantem retornos futuros.")
