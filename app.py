# -*- coding: utf-8 -*-
# ==============================================================================
# PCA CROSS-SECTIONAL MEAN REVERSION — STAT ARB DASHBOARD
#
# ESTRATÉGIA:
#   1. Universo: top-20 ações US mais líquidas (dollar volume) | preço > $5
#   2. Features: log-retornos dos últimos 60 dias de fechamento
#   3. Fator Model: PCA com 3 componentes (fatores estatísticos)
#   4. Alpha Signal: resíduo OLS padronizado (z-score idiossincrático)
#   5. Sinal: z-score < -1.5 → ação anormalmente barata → SHORT
#   6. Pesos: proporcionais à magnitude do z-score negativo
#   7. Rebalanceamento: mensal (pré-mercado)
#   8. Cash buffer: 5% livre sempre
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_OK = True
except ImportError:
    AUTOREFRESH_OK = False

# ==============================================================================
# PÁGINA
# ==============================================================================
st.set_page_config(page_title="PCA Stat Arb", page_icon="📐", layout="wide")
st.markdown("""
<style>
:root{--bg:#0b0e14;--card:#131720;--border:#2a3048;
      --text:#e8eaf6;--sub:#8b949e;
      --gold:#c9a84c;--green:#3fb950;--red:#f85149;
      --blue:#58a6ff;--purple:#bc8cff;--yellow:#d29922;}
.stApp{background:var(--bg);color:var(--text);}
section[data-testid="stSidebar"]{background:var(--card);border-right:1px solid var(--border);}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:12px;
     padding:18px 16px;text-align:center;box-shadow:0 4px 20px rgba(0,0,0,.6);}
.kpi .lbl{color:var(--sub);font-size:.72rem;text-transform:uppercase;
           letter-spacing:1px;margin-bottom:6px;}
.kpi .val{font-size:1.75rem;font-weight:700;}
.kpi .sub{font-size:.7rem;color:var(--sub);margin-top:4px;}
.stock-card{background:var(--card);border:1px solid var(--border);border-radius:10px;
            padding:14px 16px;margin:6px 0;}
.z-bar-neg{background:linear-gradient(90deg,#f85149,#2a0f0f);border-radius:4px;
           height:8px;margin-top:4px;}
.tag{display:inline-block;border-radius:5px;padding:2px 8px;
     font-size:.7rem;font-weight:700;margin-left:6px;}
.tag-short{background:#2a0f0f;color:#f85149;border:1px solid #f85149;}
.tag-hold {background:#1a1a2a;color:#8b949e;border:1px solid #2a3048;}
h1,h2,h3{color:var(--text)!important;}
.stButton>button{background:linear-gradient(135deg,#58a6ff,#bc8cff);
    color:#fff;border:none;border-radius:8px;font-weight:800;
    padding:.6rem 1.8rem;font-size:1rem;width:100%;}
div[data-testid="stMetric"]{background:var(--card);border:1px solid var(--border);
    border-radius:10px;padding:12px;}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# UNIVERSO CANDIDATO — Top US Equities por liquidez histórica
# ==============================================================================
CANDIDATE_POOL = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","JPM","V",
    "XOM","UNH","LLY","JNJ","WMT","MA","PG","HD","MRK","ORCL",
    "BAC","CVX","KO","ABBV","PEP","COST","TMO","MCD","CSCO","ABT",
    "CRM","ACN","LIN","DHR","TXN","NEE","PM","RTX","HON","UNP",
    "QCOM","IBM","GE","AMGN","LOW","BMY","SBUX","GILD","C","AMD",
    "BLK","SPGI","AXP","GS","CAT","DE","MMM","MO","DUK","SO",
    "INTC","NFLX","ADBE","PYPL","NOW","INTU","ADI","REGN","ISRG","ZTS",
]

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("## 📐 PCA Stat Arb")
    st.markdown("*Cross-Sectional Mean Reversion*")
    st.markdown("---")

    st.markdown("### 🌐 Universo")
    N_STOCKS       = st.slider("Top N ações (por dollar volume)", 10, 40, 20, 5)
    PRICE_FILTER   = st.slider("Filtro de preço mínimo ($)", 1, 20, 5, 1)
    LOOKBACK_DAYS  = st.slider("Janela PCA (dias de fechamento)", 30, 120, 60, 10)
    ANOS_BT        = st.slider("Anos de backtest", 1, 10, 5, 1)

    st.markdown("### 📐 Fator Model")
    N_COMPONENTS   = st.slider("Componentes PCA (fatores)", 1, 8, 3, 1)
    Z_ENTRY        = st.slider("Z-score threshold (SHORT)", -3.0, -0.5, -1.5, 0.1)

    st.markdown("### 💰 Portfolio")
    CAPITAL        = st.number_input("Capital (USD)", value=100_000, step=10_000)
    CASH_BUFFER    = st.slider("Cash buffer (%)", 1, 20, 5, 1) / 100
    CUSTO_BPS      = st.slider("Custo por trade (bps)", 1, 30, 10, 1) / 10_000
    SLIPPAGE_BPS   = st.slider("Slippage (bps)", 1, 20, 5, 1) / 10_000

    st.markdown("### 🔄 Refresh")
    refresh_sel = st.selectbox("Auto-refresh", ["Desligado","5 min"], index=0)
    run_btn = st.button("🚀 Executar Backtest")

if AUTOREFRESH_OK and refresh_sel != "Desligado":
    st_autorefresh(interval=300_000, key="ar")

st.markdown("# 📐 PCA Cross-Sectional Stat Arb")
st.markdown(
    "**Estratégia:** Mean reversion idiossincrática via PCA | "
    f"**Universo:** Top-{N_STOCKS} US Equities | "
    f"**Rebalanceamento:** Mensal | "
    f"**Sinal:** Z-score OLS residual < {Z_ENTRY}"
)
st.markdown("---")

# ==============================================================================
# FUNÇÕES CORE
# ==============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def baixar_precos(tickers: list, anos: int) -> pd.DataFrame:
    """Baixa fechamentos ajustados para o universo candidato."""
    start = (datetime.now() - timedelta(days=anos*365 + LOOKBACK_DAYS + 30)).strftime("%Y-%m-%d")
    raw   = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame()
    raw.columns = [str(c) for c in raw.columns]
    raw = raw.dropna(axis=1, thresh=int(len(raw)*0.85))
    return raw.sort_index()


def selecionar_universo(prices: pd.DataFrame, volumes: pd.DataFrame,
                         n: int, price_min: float, lookback: int) -> list:
    """
    Seleciona top-N ações por dollar volume médio nos últimos `lookback` dias.
    Aplica filtro de preço mínimo.
    """
    p_rec = prices.iloc[-lookback:]
    v_rec = volumes.iloc[-lookback:] if volumes is not None else None

    # Filtro de preço
    preco_atual = prices.iloc[-1]
    tickers_ok  = preco_atual[preco_atual >= price_min].index.tolist()

    if v_rec is not None:
        dv = (p_rec[tickers_ok] * v_rec[tickers_ok]).mean()
    else:
        # proxy: usa variação de preço × preço como proxy de liquidez
        dv = p_rec[tickers_ok].mean()

    dv = dv.dropna().sort_values(ascending=False)
    return dv.head(n).index.tolist()


def pca_factor_model(prices_window: pd.DataFrame, n_components: int):
    """
    Passos da estratégia:
    1. Log-transforma e de-means os preços
    2. Extrai N componentes PCA como fatores estatísticos
    3. Para cada ação, fit OLS dos retornos nos fatores
    4. Retorna resíduos padronizados (z-scores idiossincráticos)
    """
    # Log-preços e de-mean cross-seccional
    log_p  = np.log(prices_window)
    log_dm = log_p - log_p.mean(axis=1).values.reshape(-1,1)  # de-mean por dia

    # PCA nos log-preços de-meaned  (shape: T × N)
    scaler = StandardScaler()
    X_std  = scaler.fit_transform(log_dm)          # normaliza cada ação

    pca    = PCA(n_components=min(n_components, X_std.shape[1]-1, X_std.shape[0]-1))
    factors = pca.fit_transform(X_std)              # T × K

    # Para cada ação: OLS dos log-retornos nos fatores
    log_ret = log_p.diff().dropna()                 # T-1 × N
    fac_ret = pd.DataFrame(factors[1:], index=log_ret.index,
                           columns=[f'PC{i+1}' for i in range(factors.shape[1])])

    residuals   = {}
    betas       = {}
    r2_scores   = {}
    fair_values = {}

    for ticker in log_ret.columns:
        y   = log_ret[ticker].values.reshape(-1,1)
        X_f = fac_ret.values
        reg = LinearRegression().fit(X_f, y)
        y_hat = reg.predict(X_f)
        resid = y - y_hat
        residuals[ticker]   = resid.flatten()
        betas[ticker]       = reg.coef_.flatten()
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2_scores[ticker]   = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
        # fair value = log_preço explicado pelos fatores
        log_p_fv = np.log(prices_window[ticker].iloc[0]) + np.cumsum(
            np.concatenate([[0], y_hat.flatten()]))
        fair_values[ticker] = np.exp(log_p_fv)

    # Z-score do ÚLTIMO resíduo de cada ação
    z_scores = {}
    for ticker, resid in residuals.items():
        mu  = resid.mean()
        sig = resid.std()
        z_scores[ticker] = (resid[-1] - mu) / sig if sig > 0 else 0.0

    var_exp = pca.explained_variance_ratio_
    return z_scores, r2_scores, fair_values, var_exp, factors, fac_ret


def construir_portfolio(z_scores: dict, z_thr: float, capital: float,
                         cash_buffer: float) -> dict:
    """
    SHORT nas ações com z < z_thr.
    Pesos proporcionais à magnitude do z negativo,
    normalizados para gross_exposure = capital × (1 - cash_buffer).
    """
    shorts = {t: z for t, z in z_scores.items() if z < z_thr}
    if not shorts:
        return {}

    gross_exp = capital * (1 - cash_buffer)
    total_mag = sum(abs(z) for z in shorts.values())
    weights   = {t: abs(z)/total_mag * gross_exp for t, z in shorts.items()}
    return weights


def backtest_mensal(prices: pd.DataFrame, volumes: pd.DataFrame,
                     n_stocks: int, price_min: float,
                     lookback: int, n_components: int, z_thr: float,
                     capital: float, cash_buffer: float,
                     custo: float, slippage: float) -> dict:
    """
    Walk-forward mensal:
    - A cada mês re-seleciona universo + re-ajusta PCA + sinal
    - SHORT nas ações com z < z_thr
    - Hold até próximo rebalanceamento
    """
    # Datas de rebalanceamento (primeiro dia útil de cada mês)
    all_dates  = prices.index
    months     = prices.resample('MS').first().index   # início de cada mês
    months     = [m for m in months if m >= all_dates[lookback + 5]]

    cap        = capital
    equity     = [(all_dates[lookback], cap)]
    all_trades = []
    port_history = []     # snapshot mensal do portfolio
    prev_port  = {}       # posição anterior {ticker: valor_short_usd}

    bh_start   = None     # Buy & Hold benchmark (long equal-weight universe)

    for i, rebal_dt in enumerate(months[:-1]):
        next_dt = months[i+1] if i+1 < len(months) else all_dates[-1]

        # Dados disponíveis até o rebalanceamento (sem lookahead)
        hist = prices.loc[:rebal_dt]
        if len(hist) < lookback + 5:
            continue

        # Selecionar universo
        vol_hist = volumes.loc[:rebal_dt] if volumes is not None else None
        try:
            universe = selecionar_universo(hist, vol_hist, n_stocks, price_min, lookback)
        except:
            continue
        if len(universe) < n_components + 2:
            continue

        # Janela PCA: últimos `lookback` dias
        window = hist[universe].iloc[-lookback:]
        if window.isnull().any().any():
            window = window.ffill().dropna(axis=1)
            universe = window.columns.tolist()
        if len(universe) < n_components + 2:
            continue

        # Fator model
        try:
            z_scores, r2, fv, var_exp, factors, fac_ret = pca_factor_model(
                window, n_components)
        except Exception as e:
            continue

        # Portfolio
        weights = construir_portfolio(z_scores, z_thr, cap, cash_buffer)

        # ── Simular retorno no período até próximo rebalanceamento ────────────
        period_prices = prices[universe].loc[rebal_dt:next_dt]
        if len(period_prices) < 2:
            continue

        p_open  = period_prices.iloc[0]    # preço de entrada
        p_close = period_prices.iloc[-1]   # preço de saída

        period_pnl = 0.0
        trades_period = []

        for ticker, notional in weights.items():
            if ticker not in p_open.index or ticker not in p_close.index:
                continue
            pe = p_open[ticker]
            px = p_close[ticker]
            if pe <= 0 or np.isnan(pe) or np.isnan(px):
                continue

            # Shares short (notional / entry_price)
            shares   = notional / pe
            # Lucro do SHORT: shorted @ pe, coberto @ px
            pnl_raw  = shares * (pe - px)           # SHORT ganha se px < pe
            # Custos: entry + exit
            cost     = notional * (custo + slippage) * 2
            pnl_net  = pnl_raw - cost

            period_pnl += pnl_net
            trades_period.append({
                'rebal_dt': rebal_dt,
                'ticker':   ticker,
                'z_score':  z_scores.get(ticker, 0),
                'notional': notional,
                'weight_pct': notional/cap*100,
                'pe':       pe, 'px': px,
                'ret_pct':  (pe-px)/pe*100,    # retorno do short
                'pnl':      pnl_net,
            })

        cap += period_pnl
        equity.append((next_dt, cap))
        all_trades.extend(trades_period)

        # Snapshot do portfolio
        port_history.append({
            'date':       rebal_dt,
            'universe':   universe,
            'z_scores':   z_scores,
            'weights':    weights,
            'var_exp':    var_exp,
            'r2':         r2,
            'period_pnl': period_pnl,
            'n_shorts':   len(weights),
        })

        if bh_start is None and len(universe) > 0:
            bh_start = {'date': rebal_dt, 'prices': p_open.to_dict()}

    # ── Métricas ──────────────────────────────────────────────────────────────
    eq_df   = pd.DataFrame(equity, columns=['date','equity']).set_index('date')
    eq_vals = eq_df['equity'].values
    peak    = np.maximum.accumulate(eq_vals)
    dd      = (eq_vals - peak) / peak
    rets_m  = eq_df['equity'].pct_change().dropna()

    sharpe  = (rets_m.mean() / rets_m.std() * np.sqrt(12)
               if rets_m.std() > 0 else 0.0)
    sortino_dn = rets_m[rets_m < 0].std()
    sortino = (rets_m.mean() / sortino_dn * np.sqrt(12)
               if sortino_dn > 0 else 0.0)
    ret_a   = (eq_vals[-1]/capital - 1)*100
    max_dd  = float(dd.min()*100)
    calmar  = ret_a / abs(max_dd) if max_dd < 0 else 0.0

    tr_df   = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
    win_r   = float((tr_df['pnl']>0).mean()*100) if len(tr_df) else 0.0
    n_tr    = len(tr_df)
    wins    = tr_df[tr_df['pnl']>0]['pnl'].sum() if len(tr_df) else 0
    losses  = tr_df[tr_df['pnl']<0]['pnl'].sum() if len(tr_df) else 0
    pf      = wins/abs(losses) if losses != 0 else float('inf')

    # B&H equal-weight long (benchmark)
    bh_ret  = 0.0
    if bh_start:
        bh_u = list(bh_start['prices'].keys())
        try:
            bh_p_start = np.nanmean(list(bh_start['prices'].values()))
            bh_p_end   = prices[bh_u].iloc[-1].mean()
            bh_ret     = (bh_p_end/bh_p_start - 1)*100
        except:
            pass

    return {
        'eq_df':       eq_df,
        'dd':          pd.Series(dd, index=eq_df.index),
        'trades':      tr_df,
        'port_history':port_history,
        'ret_acum':    ret_a,
        'bh_ret':      bh_ret,
        'sharpe':      float(sharpe),
        'sortino':     float(sortino),
        'max_dd':      max_dd,
        'calmar':      calmar,
        'win_rate':    win_r,
        'n_trades':    n_tr,
        'cap_final':   float(eq_vals[-1]),
        'profit_factor': pf,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def baixar_volumes(tickers: list, anos: int) -> pd.DataFrame:
    start = (datetime.now() - timedelta(days=anos*365 + LOOKBACK_DAYS + 30)).strftime("%Y-%m-%d")
    raw   = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Volume"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame()
    raw.columns = [str(c) for c in raw.columns]
    return raw.sort_index()

# ==============================================================================
# EXECUÇÃO
# ==============================================================================
if run_btn or 'results' not in st.session_state:
    prog = st.progress(0, text="⬇️ Baixando preços e volumes...")
    try:
        prices  = baixar_precos(CANDIDATE_POOL, ANOS_BT)
        volumes = baixar_volumes(CANDIDATE_POOL, ANOS_BT)
        prog.progress(25, text=f"✅ {len(prices.columns)} ativos | Selecionando universo...")

        # Sinal ATUAL (snapshot do último período)
        univ_atual = selecionar_universo(prices, volumes, N_STOCKS, PRICE_FILTER, LOOKBACK_DAYS)
        window_atual = prices[univ_atual].iloc[-LOOKBACK_DAYS:]
        window_atual = window_atual.ffill().dropna(axis=1)
        univ_atual   = window_atual.columns.tolist()

        prog.progress(45, text="📐 Rodando PCA fator model...")
        z_now, r2_now, fv_now, var_exp_now, factors_now, fac_ret_now = pca_factor_model(
            window_atual, N_COMPONENTS)
        port_now = construir_portfolio(z_now, Z_ENTRY, CAPITAL, CASH_BUFFER)

        prog.progress(65, text="📅 Backtest mensal walk-forward...")
        bt = backtest_mensal(
            prices, volumes, N_STOCKS, PRICE_FILTER,
            LOOKBACK_DAYS, N_COMPONENTS, Z_ENTRY,
            CAPITAL, CASH_BUFFER, CUSTO_BPS, SLIPPAGE_BPS
        )
        prog.progress(100, text="✅ Concluído!")

        st.session_state['results'] = {
            'prices': prices, 'volumes': volumes,
            'univ_atual': univ_atual,
            'z_now': z_now, 'r2_now': r2_now, 'fv_now': fv_now,
            'var_exp_now': var_exp_now, 'factors_now': factors_now,
            'port_now': port_now, 'bt': bt,
        }
        prog.empty()
    except Exception as e:
        prog.empty()
        st.error(f"Erro: {e}")
        import traceback; st.code(traceback.format_exc())
        st.stop()

res = st.session_state.get('results')
if not res:
    st.info("Configure os parâmetros e clique em **🚀 Executar Backtest**.")
    st.stop()

bt  = res['bt']
z   = res['z_now']
pn  = res['port_now']

# ==============================================================================
# KPIs
# ==============================================================================
st.markdown("## 📊 Performance — Walk-Forward Mensal")

def kpi(col, lbl, val, css, sub=""):
    col.markdown(f"""<div class="kpi">
        <div class="lbl">{lbl}</div>
        <div class="val" style="color:{'#3fb950' if css=='green' else '#f85149' if css=='red' else '#58a6ff' if css=='blue' else '#c9a84c'};">{val}</div>
        <div class="sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)
kpi(c1,"Retorno Total",  f"{bt['ret_acum']:+.2f}%",
    "green" if bt['ret_acum']>=0 else "red", f"B&H EW Long: {bt['bh_ret']:+.2f}%")
kpi(c2,"Sharpe Ratio",   f"{bt['sharpe']:.3f}",
    "green" if bt['sharpe']>=1 else "blue" if bt['sharpe']>=0 else "red","Mensal × √12")
kpi(c3,"Win Rate",       f"{bt['win_rate']:.1f}%",
    "green" if bt['win_rate']>=50 else "red", f"{bt['n_trades']} trades")
kpi(c4,"Max Drawdown",   f"{bt['max_dd']:.2f}%","red","Pior queda")

st.markdown("<br>",unsafe_allow_html=True)
m1,m2,m3,m4,m5,m6 = st.columns(6)
m1.metric("Sortino",       f"{bt['sortino']:.3f}")
m2.metric("Calmar",        f"{bt['calmar']:.3f}")
m3.metric("Profit Factor", f"{bt['profit_factor']:.2f}" if bt['profit_factor']!=float('inf') else "∞")
m4.metric("N° Trades",     bt['n_trades'])
m5.metric("Capital Final", f"${bt['cap_final']:,.0f}")
m6.metric("Cash Buffer",   f"{CASH_BUFFER:.0%}")

st.markdown("---")

# ==============================================================================
# TABS
# ==============================================================================
BG,CARD,GRID,TXT='#0b0e14','#131720','#2a3048','#e8eaf6'

def bl(h=480):
    return dict(height=h,paper_bgcolor=BG,plot_bgcolor=CARD,
                font=dict(color=TXT),legend=dict(bgcolor=CARD),
                margin=dict(l=50,r=20,t=50,b=40))
def uf(f):
    f.update_xaxes(gridcolor=GRID,zeroline=False)
    f.update_yaxes(gridcolor=GRID,zeroline=False)
    return f

tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "🎯 Portfolio Atual","📉 Equity & Drawdown",
    "📐 PCA Fator Model","📋 Trades & Analytics","📚 Metodologia"
])

# ── Tab 1: Portfolio Atual ────────────────────────────────────────────────────
with tab1:
    st.markdown(f"### 🗓️ Portfolio — Rebalanceamento {datetime.now().strftime('%d/%m/%Y')}")
    st.markdown(f"**Universo selecionado:** {len(res['univ_atual'])} ações | "
                f"**Shorts ativos:** {len(pn)} ações | "
                f"**Gross Exposure:** ${sum(pn.values()):,.0f} "
                f"({sum(pn.values())/CAPITAL*100:.1f}% do capital)")

    # Z-scores de todo o universo
    z_df = pd.DataFrame([
        {'Ticker': t,
         'Z-Score': round(z.get(t,0),3),
         'R²': round(res['r2_now'].get(t,0),3),
         'Notional ($)': round(pn.get(t,0),2),
         'Peso (%)': round(pn.get(t,0)/CAPITAL*100,2),
         'Sinal': 'SHORT 🔴' if t in pn else ('—' if z.get(t,0) >= Z_ENTRY else 'Quase'),
         'Preço Atual': round(res['prices'][t].iloc[-1],2) if t in res['prices'].columns else 0,
        }
        for t in sorted(res['univ_atual'], key=lambda x: z.get(x,0))
    ])

    # Gráfico de barras Z-scores
    fig_z = go.Figure()
    colors_z = ['#f85149' if v < Z_ENTRY else '#2a3048' for v in z_df['Z-Score']]
    fig_z.add_trace(go.Bar(x=z_df['Ticker'], y=z_df['Z-Score'],
                            marker_color=colors_z, name="Z-Score"))
    fig_z.add_hline(y=Z_ENTRY, line_dash="dash", line_color="#d29922",
                    annotation_text=f"Threshold SHORT ({Z_ENTRY})",
                    annotation_position="top right")
    fig_z.add_hline(y=0, line_color="#2a3048")
    fig_z.update_layout(title="Z-Score Idiossincrático por Ação (Universo Atual)",
                         xaxis_title="Ação", yaxis_title="Z-Score", **bl(380))
    uf(fig_z)
    st.plotly_chart(fig_z, use_container_width=True)

    # Cards das posições SHORT
    if pn:
        st.markdown("#### 🔴 Posições SHORT Ativas")
        shorts_sorted = sorted(pn.items(), key=lambda x: z.get(x[0],0))
        cols_s = st.columns(min(4, len(shorts_sorted)))
        for i, (ticker, notional) in enumerate(shorts_sorted):
            zscore_val = z.get(ticker, 0)
            r2_val     = res['r2_now'].get(ticker, 0)
            preco_at   = res['prices'][ticker].iloc[-1] if ticker in res['prices'].columns else 0
            with cols_s[i % len(cols_s)]:
                st.markdown(f"""<div class="stock-card">
                    <div style="font-size:1.1rem;font-weight:700;color:#e8eaf6;">
                        {ticker} <span class="tag tag-short">SHORT</span>
                    </div>
                    <div style="color:#f85149;font-size:1.4rem;font-weight:700;margin:6px 0;">
                        Z = {zscore_val:.3f}
                    </div>
                    <div style="color:#8b949e;font-size:.78rem;">
                        Notional: <b style="color:#e8eaf6">${notional:,.0f}</b>
                        ({notional/CAPITAL*100:.1f}%)
                    </div>
                    <div style="color:#8b949e;font-size:.78rem;">
                        Preço: <b style="color:#e8eaf6">${preco_at:.2f}</b>
                        &nbsp;|&nbsp; R²: <b style="color:#58a6ff">{r2_val:.3f}</b>
                    </div>
                    <div class="z-bar-neg" style="width:{min(abs(zscore_val)/3*100,100):.0f}%"></div>
                </div>""", unsafe_allow_html=True)
    else:
        st.info(f"Nenhuma ação com Z-score < {Z_ENTRY} no universo atual. "
                "Tente reduzir o threshold na barra lateral.")

    st.markdown("#### 📋 Universo Completo — Z-Scores")
    st.dataframe(z_df.style.format({
        'Z-Score':'{:.3f}','R²':'{:.3f}',
        'Notional ($)':'${:,.2f}','Peso (%)':'{:.2f}%',
        'Preço Atual':'${:.2f}'
    }).background_gradient(subset=['Z-Score'], cmap='RdYlGn'),
    use_container_width=True)

# ── Tab 2: Equity & Drawdown ──────────────────────────────────────────────────
with tab2:
    eq_df = bt['eq_df']
    dd_s  = bt['dd']

    fig_e = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.65, 0.35],
                           subplot_titles=["Equity Curve (Walk-Forward Mensal)",
                                           "Drawdown (%)"])
    fig_e.add_trace(go.Scatter(x=eq_df.index, y=eq_df['equity'],
        name="Equity", fill='tozeroy',
        line=dict(color='#58a6ff',width=2.5),
        fillcolor='rgba(88,166,255,.08)'), row=1, col=1)
    fig_e.add_hline(y=CAPITAL, line_dash="dash", line_color="#8b949e",
                    annotation_text=f"Capital inicial ${CAPITAL:,.0f}", row=1, col=1)

    # Marcar rebalanceamentos
    if bt['port_history']:
        rb_dates = [p['date'] for p in bt['port_history']]
        rb_eqs   = [eq_df['equity'].asof(d) for d in rb_dates]
        fig_e.add_trace(go.Scatter(x=rb_dates, y=rb_eqs, mode='markers',
            name="Rebalanceamento",
            marker=dict(color='#d29922',size=7,symbol='diamond')), row=1, col=1)

    fig_e.add_trace(go.Scatter(x=dd_s.index, y=dd_s.values*100,
        name="Drawdown", fill='tozeroy',
        line=dict(color='#f85149',width=1.5),
        fillcolor='rgba(248,81,73,.12)'), row=2, col=1)

    fig_e.update_layout(**bl(580)); uf(fig_e)
    st.plotly_chart(fig_e, use_container_width=True)

    # Retorno mensal por rebalanceamento
    if bt['port_history']:
        ph = pd.DataFrame([{
            'Data': p['date'].strftime('%Y-%m'),
            'N Shorts': p['n_shorts'],
            'P&L ($)': round(p['period_pnl'],2),
            'P&L (%)': round(p['period_pnl']/CAPITAL*100,3),
        } for p in bt['port_history']])

        fig_mon = go.Figure(go.Bar(
            x=ph['Data'], y=ph['P&L (%)'],
            marker_color=np.where(ph['P&L (%)']>=0,'#3fb950','#f85149')))
        fig_mon.update_layout(title="P&L Mensal por Rebalanceamento (%)", **bl(320)); uf(fig_mon)
        st.plotly_chart(fig_mon, use_container_width=True)

# ── Tab 3: PCA Fator Model ────────────────────────────────────────────────────
with tab3:
    var_exp = res['var_exp_now']
    ca3,cb3 = st.columns(2)

    with ca3:
        # Scree plot
        fig_scree = go.Figure()
        n_pc = len(var_exp)
        fig_scree.add_trace(go.Bar(x=[f'PC{i+1}' for i in range(n_pc)],
                                    y=var_exp*100, name="Variância Explicada",
                                    marker_color='#58a6ff'))
        fig_scree.add_trace(go.Scatter(x=[f'PC{i+1}' for i in range(n_pc)],
                                        y=np.cumsum(var_exp)*100,
                                        name="Acumulada", mode='lines+markers',
                                        line=dict(color='#c9a84c',width=2)))
        fig_scree.update_layout(title="Scree Plot — Variância Explicada por PC", **bl(340))
        uf(fig_scree)
        st.plotly_chart(fig_scree, use_container_width=True)

        # R² por ação
        r2_s = pd.Series(res['r2_now']).sort_values(ascending=True)
        fig_r2 = go.Figure(go.Bar(x=r2_s.values, y=r2_s.index, orientation='h',
                                   marker_color='#bc8cff'))
        fig_r2.update_layout(title="R² OLS por Ação (quanto o fator explica)",
                              **bl(400)); uf(fig_r2)
        st.plotly_chart(fig_r2, use_container_width=True)

    with cb3:
        # Fatores PC1 vs PC2
        fac = res['factors_now']
        if fac.shape[1] >= 2:
            fig_pc = go.Figure(go.Scatter(
                x=fac[:,0], y=fac[:,1], mode='markers',
                marker=dict(color=np.arange(len(fac)), colorscale='Viridis',
                            size=5, showscale=True,
                            colorbar=dict(title="Dia")),
                text=[f"Dia {i}" for i in range(len(fac))]))
            fig_pc.update_layout(title="PC1 vs PC2 (60 dias de fechamento)",
                                  xaxis_title="PC1", yaxis_title="PC2", **bl(340))
            uf(fig_pc)
            st.plotly_chart(fig_pc, use_container_width=True)

        # Fair value vs preço atual (top 3 shorts)
        if pn:
            top3 = sorted(pn.keys(), key=lambda x: z.get(x,0))[:3]
            fig_fv = go.Figure()
            for ticker in top3:
                if ticker in res['fv_now']:
                    px_w = res['prices'][ticker].iloc[-LOOKBACK_DAYS:].values
                    fv_w = res['fv_now'][ticker]
                    n_w  = min(len(px_w), len(fv_w))
                    fig_fv.add_trace(go.Scatter(y=px_w[-n_w:],
                        name=f"{ticker} Preço",
                        line=dict(width=1.8)))
                    fig_fv.add_trace(go.Scatter(y=fv_w[-n_w:],
                        name=f"{ticker} Fair Value",
                        line=dict(width=1.5, dash='dash')))
            fig_fv.update_layout(
                title="Preço vs Fair Value Estimado pelo Fator (Top 3 Shorts)",
                **bl(360)); uf(fig_fv)
            st.plotly_chart(fig_fv, use_container_width=True)

# ── Tab 4: Trades & Analytics ─────────────────────────────────────────────────
with tab4:
    tr_df = bt['trades']
    if len(tr_df) == 0:
        st.warning("Nenhum trade gerado. Ajuste o threshold de Z-score.")
    else:
        ca4,cb4 = st.columns(2)
        with ca4:
            # Distribuição de retornos por trade
            fig_pnl = go.Figure(go.Histogram(x=tr_df['ret_pct'], nbinsx=40,
                marker_color='#58a6ff', opacity=.85))
            fig_pnl.add_vline(x=0, line_dash="dash", line_color="#f85149")
            avg_ret = tr_df['ret_pct'].mean()
            fig_pnl.add_vline(x=avg_ret, line_dash="dot", line_color="#3fb950",
                              annotation_text=f"Média: {avg_ret:.2f}%")
            fig_pnl.update_layout(title="Distribuição Retornos por Trade (%)",
                                   **bl(320)); uf(fig_pnl)
            st.plotly_chart(fig_pnl, use_container_width=True)

            # Win Rate por ação
            wr_by_ticker = tr_df.groupby('ticker').apply(
                lambda x: (x['pnl']>0).mean()*100).sort_values()
            fig_wr = go.Figure(go.Bar(
                x=wr_by_ticker.values, y=wr_by_ticker.index,
                orientation='h',
                marker_color=['#3fb950' if v>=50 else '#f85149'
                              for v in wr_by_ticker.values]))
            fig_wr.add_vline(x=50, line_dash="dash", line_color="#8b949e")
            fig_wr.update_layout(title="Win Rate por Ação (%)", **bl(380)); uf(fig_wr)
            st.plotly_chart(fig_wr, use_container_width=True)

        with cb4:
            # P&L acumulado por ação
            pnl_by_ticker = tr_df.groupby('ticker')['pnl'].sum().sort_values()
            fig_pt = go.Figure(go.Bar(
                x=pnl_by_ticker.values, y=pnl_by_ticker.index,
                orientation='h',
                marker_color=['#3fb950' if v>=0 else '#f85149'
                              for v in pnl_by_ticker.values]))
            fig_pt.update_layout(title="P&L Total por Ação ($)", **bl(380)); uf(fig_pt)
            st.plotly_chart(fig_pt, use_container_width=True)

            # Z-score médio de entrada vs retorno
            z_vs_ret = tr_df.groupby('ticker').agg(
                z_mean=('z_score','mean'), ret_mean=('ret_pct','mean')).reset_index()
            fig_zr = go.Figure(go.Scatter(
                x=z_vs_ret['z_mean'], y=z_vs_ret['ret_mean'],
                mode='markers+text', text=z_vs_ret['ticker'],
                textposition='top center',
                marker=dict(color='#58a6ff', size=10,
                            line=dict(color='#bc8cff',width=1))))
            fig_zr.add_vline(x=Z_ENTRY, line_dash="dash", line_color="#d29922")
            fig_zr.add_hline(y=0, line_color="#2a3048")
            fig_zr.update_layout(title="Z-Score de Entrada vs Retorno Médio (%)",
                                  xaxis_title="Z-Score Médio",
                                  yaxis_title="Retorno Médio (%)", **bl(380))
            uf(fig_zr)
            st.plotly_chart(fig_zr, use_container_width=True)

        st.markdown("#### 📋 Histórico de Trades")
        st.dataframe(tr_df.sort_values('rebal_dt',ascending=False)
                     .style.format({
                         'z_score':'{:.3f}','notional':'${:,.0f}',
                         'weight_pct':'{:.2f}%','pe':'${:.2f}','px':'${:.2f}',
                         'ret_pct':'{:+.3f}%','pnl':'${:+,.2f}',
                     }).background_gradient(subset=['ret_pct'],cmap='RdYlGn'),
                     use_container_width=True)

# ── Tab 5: Metodologia ────────────────────────────────────────────────────────
with tab5:
    st.markdown("""
    ### 📚 Metodologia — PCA Cross-Sectional Mean Reversion Stat Arb

    ---

    #### 1. Universo
    A cada rebalanceamento mensal, seleciona as **top-N ações US** por dollar volume médio
    nos últimos 60 dias. Aplica filtro de preço mínimo (padrão $5) para excluir penny stocks
    e garantir liquidez real de execução.

    ---

    #### 2. Fator Model via PCA
    Com os últimos **60 fechamentos diários**:
    1. **Log-transforma** os preços: `log(P)`
    2. **De-means cross-seccional**: subtrai a média do mercado a cada dia
    3. Aplica **PCA** → extrai os **K primeiros componentes** como *fatores estatísticos*
       (análogos a fatores de risco de mercado, setor, momentum)
    4. Para cada ação, estima OLS: `log_ret_i = β₁·PC1 + β₂·PC2 + β₃·PC3 + ε_i`

    ---

    #### 3. Alpha Signal
    O **resíduo OLS padronizado (z-score)** captura o componente idiossincrático:
    ```
    z_i = (ε_i,t - μ_ε) / σ_ε
    ```
    - **z < -1.5** → ação anormalmente *barata* vs o fator model → espera-se mean reversion *para cima*
    - Como a estratégia é **short-only** neste framework, ela faz SHORT nas ações com z < threshold
      (apostando que o preço vai reverter para o fair value — ou seja, *cair menos* que o mercado)

    ---

    #### 4. Portfolio Construction
    - **Pesos** proporcionais à magnitude do z negativo: `w_i ∝ |z_i|`
    - Normalizados para **gross exposure = capital × (1 - cash_buffer)**
    - **Sem long leg** explícita, **sem stop-loss**, **sem saída por tempo** além do rebalanceamento

    ---

    #### 5. Rebalanceamento
    - **Mensal**, pré-mercado (decisão baseada em fechamentos anteriores)
    - Universo e pesos são **completamente recalculados** a cada mês
    - Custos aplicados: spread + comissão na entrada e saída

    ---

    #### ⚠️ Considerações Importantes
    - A estratégia é **SHORT-ONLY**: performa melhor em mercados de alta volatilidade
      ou correções; underperforma em bull markets estruturais
    - Para stat arb completo, normalmente se adiciona uma **long leg** nos nomes com z > +1.5
      (market neutral)
    - Os resíduos OLS são **não-estacionários por construção** em séries de preço;
      o sinal é mais robusto em log-retornos (implementado aqui)
    """)

st.markdown("---")
with st.expander("🗂️ Preços brutos (últimas 30 linhas)"):
    st.dataframe(res['prices'][res['univ_atual']].tail(30).style.format("${:.2f}"),
                 use_container_width=True)

st.caption("⚠️ Dashboard educacional. Não constitui recomendação de investimento.")
