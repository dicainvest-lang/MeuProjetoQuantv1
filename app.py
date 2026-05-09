# -*- coding: utf-8 -*-
# ==============================================================================
# PCA STAT ARBITRAGE — STREAMLIT PORT OF QUANTCONNECT LEAN ALGORITHM
#
# Replica fiel de PcaStatArbitrageAlgorithm:
#   • Universo: top-20 ações US por dollar_volume, preço > $5
#   • Dados: 60 dias de fechamentos diários, dropna(axis=1)
#   • PCA fit em log-preços centrados (column-wise)
#   • Factors = sample @ pca.components_.T  →  primeiros N_COMPONENTS
#   • OLS por ação com constante (sm.add_constant)
#   • Z-score do resíduo no último dia
#   • Sinal: z < ENTRY_Z_SCORE  →  SHORT
#   • Peso: |z_i| / sum(|z_j|)  (short → peso negativo no portfolio)
#   • Rebalanceamento: mensal (início de mês), pré-mercado 08:00
#   • Free cash buffer: 5 %
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
import statsmodels.api as sm
from sklearn.decomposition import PCA

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_OK = True
except ImportError:
    AUTOREFRESH_OK = False

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(page_title="PCA Stat Arb · QC Port", page_icon="📐", layout="wide")
st.markdown("""
<style>
:root{--bg:#0b0e14;--card:#131720;--border:#2a3048;
      --text:#e8eaf6;--sub:#8b949e;
      --green:#3fb950;--red:#f85149;--blue:#58a6ff;
      --purple:#bc8cff;--yellow:#d29922;--gold:#c9a84c;}
.stApp{background:var(--bg);color:var(--text);}
section[data-testid="stSidebar"]{background:var(--card);
  border-right:1px solid var(--border);}
.kpi{background:var(--card);border:1px solid var(--border);
     border-radius:12px;padding:18px 16px;text-align:center;
     box-shadow:0 4px 20px rgba(0,0,0,.55);}
.kpi .lbl{color:var(--sub);font-size:.72rem;text-transform:uppercase;
           letter-spacing:1px;margin-bottom:5px;}
.kpi .val{font-size:1.75rem;font-weight:700;}
.kpi .sub{font-size:.7rem;color:var(--sub);margin-top:4px;}
.scard{background:var(--card);border:1px solid var(--border);
       border-radius:10px;padding:13px 15px;margin:5px 0;}
.tag-short{display:inline-block;background:#2a0f0f;color:#f85149;
           border:1px solid #f85149;border-radius:5px;
           padding:2px 7px;font-size:.68rem;font-weight:700;}
h1,h2,h3{color:var(--text)!important;}
.stButton>button{background:linear-gradient(135deg,#58a6ff,#bc8cff);
  color:#fff;border:none;border-radius:8px;font-weight:800;
  padding:.6rem 1.8rem;font-size:1rem;width:100%;}
div[data-testid="stMetric"]{background:var(--card);
  border:1px solid var(--border);border-radius:10px;padding:12px;}
code{background:#1a1f2e;border-radius:4px;padding:1px 5px;}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CANDIDATE POOL  (amplo o suficiente para filtrar top-20 por dollar vol)
# ==============================================================================
CANDIDATE_POOL = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","TSLA","AVGO","JPM",
    "V","XOM","UNH","LLY","JNJ","WMT","MA","PG","HD","MRK","ORCL","BAC",
    "CVX","KO","ABBV","PEP","COST","TMO","MCD","CSCO","ABT","CRM","ACN",
    "LIN","DHR","TXN","NEE","PM","RTX","HON","UNP","QCOM","IBM","GE","AMGN",
    "LOW","BMY","SBUX","GILD","C","AMD","BLK","SPGI","AXP","GS","CAT","DE",
    "INTC","NFLX","ADBE","NOW","INTU","ADI","REGN","ISRG","ZTS","MMM","MO",
    "DUK","SO","F","GM","UBER","COIN","PLTR","ARM","SHOP","SQ","SNOW","AMAT",
]

# ==============================================================================
# SIDEBAR — parâmetros espelhados no LEAN algorithm
# ==============================================================================
with st.sidebar:
    st.markdown("## 📐 PCA Stat Arb")
    st.markdown("*QuantConnect LEAN port*")
    st.markdown("---")

    st.markdown("### ⚙️ Parâmetros do Algoritmo")
    LOOKBACK_DAYS  = st.slider("_lookback_days",       30, 120, 60,  10)
    PCA_COMPONENTS = st.slider("_pca_components",       1,   8,  3,   1)
    ENTRY_Z_SCORE  = st.slider("_entry_z_score",      -3.0,-0.5,-1.5, 0.1)
    N_UNIVERSE     = st.slider("Universo (top N)",     10,  40, 20,   5)
    PRICE_FILTER   = st.slider("Filtro preço ($)",      1,  20,  5,   1)
    ANOS_BT        = st.slider("Período backtest (anos)", 1, 10,  5,   1)

    st.markdown("### 💰 Capital & Custos")
    CAPITAL        = st.number_input("set_cash ($)", value=100_000, step=10_000)
    FREE_CASH_PCT  = st.slider("free_portfolio_value_percentage (%)", 1, 20, 5, 1) / 100
    CUSTO_BPS      = st.slider("Custo entrada+saída (bps)", 0, 30, 10, 1) / 10_000
    SLIPPAGE_BPS   = st.slider("Slippage (bps)",            0, 20,  5, 1) / 10_000

    st.markdown("### 🔄 Refresh")
    refresh_sel = st.selectbox("Auto-refresh", ["Desligado","5 min"], index=0)
    run_btn = st.button("🚀 Executar Backtest")

if AUTOREFRESH_OK and refresh_sel != "Desligado":
    st_autorefresh(interval=300_000, key="ar")

st.markdown("# 📐 PCA Cross-Sectional Stat Arb")
st.markdown(
    f"QuantConnect `PcaStatArbitrageAlgorithm` · "
    f"**Lookback:** {LOOKBACK_DAYS}d · **PCA:** {PCA_COMPONENTS} componentes · "
    f"**Entry z:** {ENTRY_Z_SCORE} · **Universo:** Top-{N_UNIVERSE} · "
    f"**Rebalanceamento:** mensal"
)
st.markdown("---")

# ==============================================================================
# DOWNLOAD
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def _baixar(tickers, anos, extra_days=0):
    start = (datetime.now() -
             timedelta(days=anos*365 + LOOKBACK_DAYS + extra_days + 30)
             ).strftime("%Y-%m-%d")
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    close  = raw["Close"]  if "Close"  in raw else raw.xs("Close",  axis=1, level=0)
    volume = raw["Volume"] if "Volume" in raw else raw.xs("Volume", axis=1, level=0)
    if isinstance(close,  pd.Series): close  = close.to_frame()
    if isinstance(volume, pd.Series): volume = volume.to_frame()
    close.columns  = [str(c) for c in close.columns]
    volume.columns = [str(c) for c in volume.columns]
    return close.sort_index(), volume.sort_index()

# ==============================================================================
# REPLICA EXATA DE _compute_weights_by_symbol()
# ==============================================================================
def compute_weights_by_symbol(history_close: pd.DataFrame,
                               pca_components: int,
                               entry_z: float) -> dict:
    """
    Replica linha-a-linha o método LEAN:

        sample  = np.log(history.dropna(axis=1))
        sample  = sample - sample.mean()          # centrar column-wise
        pca     = PCA().fit(sample)
        factors = sample @ pca.components_.T  →  [:, :n_components]
        factors = sm.add_constant(factors)
        ols     = sm.OLS(sample[col], factors).fit()
        resid   = model.resid
        z_score = (resid - resid.mean()) / resid.std()   [último dia]
        selecionados: z < entry_z
        peso: z_i / sum(|z_j|)   →  negativo (SHORT)
    """
    # --- passo 1: log + dropna(axis=1) + centrar ---
    sample = np.log(history_close.dropna(axis=1))
    sample = sample - sample.mean()                      # column-wise

    if sample.shape[1] < pca_components + 1:
        return {}, pd.Series(dtype=float), pd.Series(dtype=float), None, None

    # --- passo 2: PCA fit ---
    pca_model = PCA().fit(sample)

    # --- passo 3: factors = sample @ components.T  →  [:, :n] ---
    factors_raw = np.dot(sample, pca_model.components_.T)[:, :pca_components]

    # --- passo 4: add constant (intercept) ---
    factors = sm.add_constant(factors_raw)

    # --- passo 5: OLS por symbol ---
    ols_models = {
        col: sm.OLS(sample[col], factors).fit()
        for col in sample.columns
    }

    # --- passo 6: residuals DataFrame ---
    residuals = pd.DataFrame(
        {col: m.resid for col, m in ols_models.items()},
        index=sample.index
    )

    # --- passo 7: z-scores do último dia ---
    z_scores = ((residuals - residuals.mean()) / residuals.std()).iloc[-1]

    # --- passo 8: selecionar & normalizar ---
    selected = z_scores[z_scores < entry_z]
    denom    = selected.abs().sum()
    if denom == 0:
        return {}, z_scores, residuals, pca_model, factors_raw

    weights = selected * (1.0 / denom)          # valores negativos (SHORT)
    return weights.to_dict(), z_scores, residuals, pca_model, factors_raw


# ==============================================================================
# SELECIONAR UNIVERSO (replica o lambda do add_universe)
# ==============================================================================
def selecionar_universo(close_hist, volume_hist, as_of_date,
                         n, price_min, lookback):
    """
    Replica:
        sorted((f for f in fundamentals if f.price > 5),
               key=lambda f: f.dollar_volume)[-20:]
    Proxy: dollar_volume = preço × volume (média dos últimos `lookback` dias)
    """
    idx = close_hist.index
    pos = idx.get_indexer([as_of_date], method='pad')[0]
    if pos < lookback:
        return []
    c_w = close_hist.iloc[max(0, pos-lookback):pos+1]
    v_w = volume_hist.iloc[max(0, pos-lookback):pos+1]

    # Filtro de preço
    preco_atual = c_w.iloc[-1]
    ok = preco_atual[preco_atual >= price_min].index.tolist()

    # Dollar volume proxy
    dv = (c_w[ok] * v_w[ok]).mean().dropna()
    top = dv.nlargest(n).index.tolist()
    return top


# ==============================================================================
# BACKTEST WALK-FORWARD MENSAL (replica on_warmup_finished + _rebalance)
# ==============================================================================
def backtest(close, volume, n_universe, price_min, lookback,
             pca_comp, entry_z, capital, free_cash, custo, slippage):
    """
    Walk-forward mensal:
    • A cada início de mês re-seleciona universo + recalcula pesos
    • SHORT com peso = |z_i| / Σ|z_j|, scaled por gross_exposure
    • Hold até próximo rebalanceamento
    • Aplica custos de entrada + saída
    """
    # Warmup: precisamos de `lookback` dias antes de começar
    all_dates = close.index
    rebal_dates = (
        close.resample('MS').first()
             .index
             .intersection(all_dates)
    )
    rebal_dates = [d for d in rebal_dates if
                   all_dates.get_loc(d) >= lookback + 5]

    cap         = capital
    equity      = [(all_dates[lookback], cap)]
    all_trades  = []
    snapshots   = []         # um por rebalanceamento

    for ri, rd in enumerate(rebal_dates[:-1]):
        next_rd = rebal_dates[ri + 1]

        # universo neste rebalanceamento
        universe = selecionar_universo(close, volume, rd,
                                       n_universe, price_min, lookback)
        if not universe:
            continue

        # janela de histórico disponível até rd (inclusive)
        pos     = all_dates.get_indexer([rd], method='pad')[0]
        hist_w  = close[universe].iloc[max(0, pos-lookback+1):pos+1]

        if len(hist_w) < lookback * 0.8:
            continue

        # ---- replica _compute_weights_by_symbol ----
        weights, z_scores, residuals, pca_obj, factors_arr = \
            compute_weights_by_symbol(hist_w, pca_comp, entry_z)

        if not weights:
            snapshots.append({'date': rd, 'universe': universe,
                              'z_scores': z_scores, 'weights': {},
                              'n_shorts': 0, 'period_pnl': 0.0,
                              'pca': pca_obj, 'residuals': residuals,
                              'factors': factors_arr})
            continue

        # preços no período de holding
        pos_next = all_dates.get_indexer([next_rd], method='pad')[0]
        period_p = close[universe].iloc[pos:pos_next+1]
        if len(period_p) < 2:
            continue

        p_entry = period_p.iloc[0]
        p_exit  = period_p.iloc[-1]

        gross_exp = cap * (1 - free_cash)
        period_pnl = 0.0
        trades_now  = []

        for ticker, w in weights.items():                # w < 0 (SHORT)
            if ticker not in p_entry.index: continue
            pe = p_entry[ticker]; px = p_exit[ticker]
            if pe <= 0 or np.isnan(pe) or np.isnan(px): continue

            notional = abs(w) * gross_exp               # $ alocado
            shares   = notional / pe
            # SHORT pnl: ganho se px < pe
            pnl_raw  = shares * (pe - px)
            cost     = notional * (custo + slippage) * 2
            pnl_net  = pnl_raw - cost

            period_pnl += pnl_net
            trades_now.append({
                'rebal_dt':   rd,
                'ticker':     ticker,
                'z_score':    float(z_scores.get(ticker, 0)),
                'weight_pct': float(abs(w) * 100),
                'notional':   notional,
                'pe': pe, 'px': px,
                'ret_short_pct': float((pe - px) / pe * 100),
                'pnl': pnl_net,
            })

        cap += period_pnl
        equity.append((next_rd, cap))
        all_trades.extend(trades_now)
        snapshots.append({
            'date':       rd,
            'universe':   universe,
            'z_scores':   z_scores,
            'weights':    weights,
            'n_shorts':   len(weights),
            'period_pnl': period_pnl,
            'pca':        pca_obj,
            'residuals':  residuals,
            'factors':    factors_arr,
        })

    # ── métricas ──────────────────────────────────────────────────────────────
    eq_df  = pd.DataFrame(equity, columns=['date','equity']).set_index('date')
    eq_v   = eq_df['equity'].values
    peak   = np.maximum.accumulate(eq_v)
    dd     = (eq_v - peak) / peak
    rets_m = eq_df['equity'].pct_change().dropna()

    def ann(s, freq=12):
        return (s.mean() / s.std() * np.sqrt(freq)) if s.std() > 0 else 0.0

    sharpe  = ann(rets_m)
    dn      = rets_m[rets_m < 0]
    sortino = (rets_m.mean() / dn.std() * np.sqrt(12)) if len(dn) > 0 and dn.std() > 0 else 0.0
    ret_a   = (eq_v[-1] / capital - 1) * 100
    max_dd  = float(dd.min() * 100)
    calmar  = ret_a / abs(max_dd) if max_dd < 0 else 0.0

    tr_df  = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
    win_r  = float((tr_df['pnl'] > 0).mean() * 100) if len(tr_df) else 0.0
    wins   = tr_df[tr_df['pnl'] > 0]['pnl'].sum() if len(tr_df) else 0
    losses = tr_df[tr_df['pnl'] < 0]['pnl'].sum() if len(tr_df) else 0
    pf     = wins / abs(losses) if losses != 0 else float('inf')

    # B&H equal-weight long benchmark (primeiros ativos do universo inicial)
    bh_ret = 0.0
    if snapshots:
        bh_u = snapshots[0]['universe'][:10]
        try:
            bh_ret = close[bh_u].iloc[-1].mean() / close[bh_u].iloc[lookback].mean() - 1
            bh_ret *= 100
        except Exception:
            pass

    return {
        'eq_df':      eq_df,
        'dd':         pd.Series(dd, index=eq_df.index),
        'trades':     tr_df,
        'snapshots':  snapshots,
        'ret_acum':   ret_a,
        'bh_ret':     bh_ret,
        'sharpe':     float(sharpe),
        'sortino':    float(sortino),
        'max_dd':     max_dd,
        'calmar':     calmar,
        'win_rate':   win_r,
        'n_trades':   len(tr_df),
        'cap_final':  float(eq_v[-1]),
        'profit_factor': pf,
    }


# ==============================================================================
# SNAPSHOT DO SINAL ATUAL (replica _compute_weights_by_symbol ao vivo)
# ==============================================================================
def sinal_atual(close, volume, n_universe, price_min,
                lookback, pca_comp, entry_z):
    today    = close.index[-1]
    universe = selecionar_universo(close, volume, today,
                                   n_universe, price_min, lookback)
    if not universe:
        return {}, pd.Series(dtype=float), universe, None, None, pd.DataFrame()

    hist_w = close[universe].iloc[-lookback:]
    weights, z_scores, residuals, pca_obj, factors_arr = \
        compute_weights_by_symbol(hist_w, pca_comp, entry_z)
    return weights, z_scores, universe, pca_obj, factors_arr, residuals


# ==============================================================================
# EXECUÇÃO
# ==============================================================================
if run_btn or 'results' not in st.session_state:
    prog = st.progress(0, text="⬇️ Baixando preços & volumes…")
    try:
        close, volume = _baixar(CANDIDATE_POOL, ANOS_BT)
        prog.progress(25, text=f"✅ {len(close.columns)} ativos baixados · Calculando sinal atual…")

        w_now, z_now, univ_now, pca_now, fac_now, resid_now = sinal_atual(
            close, volume, N_UNIVERSE, PRICE_FILTER,
            LOOKBACK_DAYS, PCA_COMPONENTS, ENTRY_Z_SCORE
        )
        prog.progress(50, text="📅 Rodando backtest walk-forward mensal…")

        bt = backtest(
            close, volume, N_UNIVERSE, PRICE_FILTER,
            LOOKBACK_DAYS, PCA_COMPONENTS, ENTRY_Z_SCORE,
            CAPITAL, FREE_CASH_PCT, CUSTO_BPS, SLIPPAGE_BPS
        )
        prog.progress(100, text="✅ Concluído!")
        st.session_state['results'] = {
            'close': close, 'volume': volume,
            'w_now': w_now, 'z_now': z_now,
            'univ_now': univ_now,
            'pca_now': pca_now, 'fac_now': fac_now, 'resid_now': resid_now,
            'bt': bt,
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

bt   = res['bt']
z    = res['z_now']
w    = res['w_now']
univ = res['univ_now']

# ==============================================================================
# KPIs
# ==============================================================================
st.markdown("## 📊 Performance — Walk-Forward Mensal")

def kpi(col, lbl, val, color, sub=""):
    col.markdown(f"""<div class="kpi">
        <div class="lbl">{lbl}</div>
        <div class="val" style="color:{color};">{val}</div>
        <div class="sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)
kpi(c1,"Retorno Total",  f"{bt['ret_acum']:+.2f}%",
    "#3fb950" if bt['ret_acum']>=0 else "#f85149",
    f"B&H EW Long: {bt['bh_ret']:+.2f}%")
kpi(c2,"Sharpe Ratio",   f"{bt['sharpe']:.3f}",
    "#3fb950" if bt['sharpe']>=1 else "#58a6ff" if bt['sharpe']>=0 else "#f85149",
    "Mensal × √12")
kpi(c3,"Win Rate",       f"{bt['win_rate']:.1f}%",
    "#3fb950" if bt['win_rate']>=50 else "#f85149",
    f"{bt['n_trades']} trades")
kpi(c4,"Max Drawdown",   f"{bt['max_dd']:.2f}%","#f85149","Pior queda")

st.markdown("<br>", unsafe_allow_html=True)
m1,m2,m3,m4,m5,m6 = st.columns(6)
m1.metric("Sortino",       f"{bt['sortino']:.3f}")
m2.metric("Calmar",        f"{bt['calmar']:.3f}")
m3.metric("Profit Factor", f"{bt['profit_factor']:.2f}" if bt['profit_factor']!=float('inf') else "∞")
m4.metric("N° Trades",     bt['n_trades'])
m5.metric("Capital Final", f"${bt['cap_final']:,.0f}")
m6.metric("Cash Buffer",   f"{FREE_CASH_PCT:.0%}")

st.markdown("---")

# ==============================================================================
# TABS
# ==============================================================================
BG,CARD,GRID,TXT = '#0b0e14','#131720','#2a3048','#e8eaf6'

def bl(h=480):
    return dict(height=h, paper_bgcolor=BG, plot_bgcolor=CARD,
                font=dict(color=TXT), legend=dict(bgcolor=CARD),
                margin=dict(l=50,r=20,t=50,b=40))
def uf(f):
    f.update_xaxes(gridcolor=GRID, zeroline=False)
    f.update_yaxes(gridcolor=GRID, zeroline=False)
    return f

tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "🎯 Sinal Atual","📉 Equity & Drawdown",
    "📐 PCA & Resíduos","📋 Trades","📚 Código LEAN"
])

# ── Tab 1: Sinal Atual ────────────────────────────────────────────────────────
with tab1:
    st.markdown(f"### 🗓️ {datetime.now().strftime('%d/%m/%Y')} — Próximo Rebalanceamento")
    st.markdown(
        f"**Universo selecionado:** {len(univ)} ações | "
        f"**Shorts ativos:** {len(w)} | "
        f"**Gross exposure:** ${sum(abs(v)*CAPITAL*(1-FREE_CASH_PCT) for v in w.values()):,.0f} "
        f"({(1-FREE_CASH_PCT)*100:.0f}% do capital)"
    )

    if not z.empty:
        # Barras de z-score para todo o universo
        z_sorted = z.sort_values()
        colors_z = ['#f85149' if v < ENTRY_Z_SCORE else '#2a3048'
                    for v in z_sorted.values]
        fig_z = go.Figure(go.Bar(
            x=z_sorted.index, y=z_sorted.values,
            marker_color=colors_z, name="Z-Score"))
        fig_z.add_hline(y=ENTRY_Z_SCORE, line_dash="dash",
                        line_color="#d29922",
                        annotation_text=f"entry_z_score = {ENTRY_Z_SCORE}",
                        annotation_position="top right")
        fig_z.add_hline(y=0, line_color="#2a3048")
        fig_z.update_layout(
            title="Z-Score Idiossincrático — Resíduo OLS Padronizado (Último Dia)",
            xaxis_title="Ticker", yaxis_title="Z-Score", **bl(380))
        uf(fig_z)
        st.plotly_chart(fig_z, use_container_width=True)

    # Cards das posições SHORT
    if w:
        st.markdown("#### 🔴 Posições SHORT — `set_holdings(targets, True)`")
        shorts_sorted = sorted(w.items(), key=lambda x: x[1])  # mais negativo primeiro
        cols_s = st.columns(min(5, len(shorts_sorted)))
        for i, (ticker, weight) in enumerate(shorts_sorted):
            z_val      = float(z.get(ticker, 0))
            notional   = abs(weight) * CAPITAL * (1 - FREE_CASH_PCT)
            preco_at   = res['close'][ticker].iloc[-1] if ticker in res['close'].columns else 0
            with cols_s[i % len(cols_s)]:
                st.markdown(f"""<div class="scard">
                    <div style="font-size:1rem;font-weight:700;color:#e8eaf6;">
                        {ticker} <span class="tag-short">SHORT</span>
                    </div>
                    <div style="color:#f85149;font-size:1.35rem;
                                font-weight:700;margin:5px 0;">
                        z = {z_val:.3f}
                    </div>
                    <div style="color:#8b949e;font-size:.76rem;">
                        peso: <b style="color:#e8eaf6">{abs(weight)*100:.1f}%</b>
                        → <b style="color:#58a6ff">${notional:,.0f}</b>
                    </div>
                    <div style="color:#8b949e;font-size:.76rem;">
                        preço: <b style="color:#e8eaf6">${preco_at:.2f}</b>
                    </div>
                    <div style="background:linear-gradient(90deg,#f85149,#2a0f0f);
                                border-radius:4px;height:6px;margin-top:6px;
                                width:{min(abs(z_val)/3*100,100):.0f}%"></div>
                </div>""", unsafe_allow_html=True)
    else:
        st.info(f"Nenhuma ação com z-score < {ENTRY_Z_SCORE} hoje. "
                "Reduza o threshold ou aumente o universo.")

    # Tabela universo completo
    st.markdown("#### 📋 Universo Atual — Z-Scores Completos")
    z_df = pd.DataFrame([{
        'Ticker':      t,
        'Z-Score':     round(float(z.get(t,0)), 4),
        'Peso Short %':round(abs(w.get(t,0))*100, 2) if t in w else 0.0,
        'Notional $':  round(abs(w.get(t,0))*CAPITAL*(1-FREE_CASH_PCT),0) if t in w else 0.0,
        'Preço $':     round(res['close'][t].iloc[-1],2) if t in res['close'].columns else 0,
        'Sinal':       'SHORT 🔴' if t in w else '—',
    } for t in sorted(univ, key=lambda x: float(z.get(x,0)))])

    st.dataframe(
        z_df.style.format({
            'Z-Score':'{:.4f}','Peso Short %':'{:.2f}%',
            'Notional $':'${:,.0f}','Preço $':'${:.2f}'
        }).background_gradient(subset=['Z-Score'], cmap='RdYlGn', vmin=-3, vmax=3),
        use_container_width=True
    )

# ── Tab 2: Equity & Drawdown ──────────────────────────────────────────────────
with tab2:
    eq_df = bt['eq_df']
    dd_s  = bt['dd']

    fig_e = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.65, 0.35],
                           subplot_titles=["Equity Curve — Walk-Forward Mensal",
                                           "Drawdown (%)"])
    fig_e.add_trace(go.Scatter(
        x=eq_df.index, y=eq_df['equity'], name="Strategy",
        fill='tozeroy', line=dict(color='#58a6ff', width=2.5),
        fillcolor='rgba(88,166,255,.08)'), row=1, col=1)
    fig_e.add_hline(y=CAPITAL, line_dash="dash", line_color="#8b949e",
                    annotation_text=f"Capital inicial", row=1, col=1)

    if bt['snapshots']:
        rb_d = [s['date'] for s in bt['snapshots']]
        rb_e = [eq_df['equity'].asof(d) for d in rb_d]
        fig_e.add_trace(go.Scatter(
            x=rb_d, y=rb_e, mode='markers', name="Rebalanceamento",
            marker=dict(color='#d29922', size=7, symbol='diamond')), row=1, col=1)

    fig_e.add_trace(go.Scatter(
        x=dd_s.index, y=dd_s.values*100, name="Drawdown",
        fill='tozeroy', line=dict(color='#f85149', width=1.5),
        fillcolor='rgba(248,81,73,.12)'), row=2, col=1)
    fig_e.update_layout(**bl(560)); uf(fig_e)
    st.plotly_chart(fig_e, use_container_width=True)

    # P&L por rebalanceamento
    if bt['snapshots']:
        ph = pd.DataFrame([{
            'Mês':     s['date'].strftime('%Y-%m'),
            'N Shorts':s['n_shorts'],
            'P&L ($)': round(s['period_pnl'],2),
            'P&L (%)': round(s['period_pnl']/CAPITAL*100,3),
        } for s in bt['snapshots']])

        fig_mon = go.Figure(go.Bar(
            x=ph['Mês'], y=ph['P&L (%)'],
            marker_color=np.where(ph['P&L (%)']>=0,'#3fb950','#f85149')))
        fig_mon.update_layout(title="P&L por Rebalanceamento Mensal (%)",**bl(300))
        uf(fig_mon)
        st.plotly_chart(fig_mon, use_container_width=True)

        st.dataframe(ph.style.format({'P&L ($)':'${:,.2f}','P&L (%)':'{:+.3f}%'}),
                     use_container_width=True)

# ── Tab 3: PCA & Resíduos ─────────────────────────────────────────────────────
with tab3:
    ca3, cb3 = st.columns(2)

    with ca3:
        # Scree plot da PCA atual
        if res['pca_now'] is not None:
            pca_obj  = res['pca_now']
            var_exp  = pca_obj.explained_variance_ratio_
            n_show   = min(15, len(var_exp))
            fig_scr  = go.Figure()
            fig_scr.add_trace(go.Bar(
                x=[f'PC{i+1}' for i in range(n_show)],
                y=var_exp[:n_show]*100, name="Variância",
                marker_color='#58a6ff'))
            fig_scr.add_trace(go.Scatter(
                x=[f'PC{i+1}' for i in range(n_show)],
                y=np.cumsum(var_exp[:n_show])*100,
                name="Acumulada", mode='lines+markers',
                line=dict(color='#c9a84c', width=2)))
            fig_scr.add_hline(y=80, line_dash="dot", line_color="#8b949e",
                              annotation_text="80%")
            fig_scr.update_layout(title="Scree Plot — PCA Atual", **bl(350))
            uf(fig_scr)
            st.plotly_chart(fig_scr, use_container_width=True)

        # Residuals — heatmap última semana
        if not res['resid_now'].empty:
            r = res['resid_now'].tail(20)
            fig_h = go.Figure(go.Heatmap(
                z=r.values.T,
                x=[d.strftime('%m-%d') for d in r.index],
                y=r.columns.tolist(),
                colorscale='RdYlGn', zmid=0,
                colorbar=dict(title="Resíduo")))
            fig_h.update_layout(
                title="Resíduos OLS — Últimos 20 Dias (Heatmap)",
                **bl(420))
            st.plotly_chart(fig_h, use_container_width=True)

    with cb3:
        # Fatores PC1 vs PC2
        if res['fac_now'] is not None and res['fac_now'].shape[1] >= 2:
            fac = res['fac_now']
            fig_pc = go.Figure(go.Scatter(
                x=fac[:,0], y=fac[:,1], mode='markers',
                marker=dict(
                    color=np.arange(len(fac)),
                    colorscale='Viridis', size=6, showscale=True,
                    colorbar=dict(title="Dia (0=mais antigo)")),
                text=[f"Dia {i}" for i in range(len(fac))]))
            fig_pc.update_layout(
                title=f"PC1 vs PC2 ({LOOKBACK_DAYS} dias)",
                xaxis_title="PC1 — Fator de Mercado",
                yaxis_title="PC2 — Fator Setorial", **bl(360))
            uf(fig_pc)
            st.plotly_chart(fig_pc, use_container_width=True)

        # Z-scores ao longo do tempo para as posições SHORT atuais
        if w and not res['resid_now'].empty:
            r   = res['resid_now']
            zts = ((r - r.mean()) / r.std())
            top_s = sorted(w.keys(), key=lambda x: float(z.get(x,0)))[:5]
            fig_zt = go.Figure()
            for t in top_s:
                if t in zts.columns:
                    fig_zt.add_trace(go.Scatter(
                        x=zts.index, y=zts[t], name=t,
                        line=dict(width=1.8)))
            fig_zt.add_hline(y=ENTRY_Z_SCORE, line_dash="dash",
                             line_color="#d29922",
                             annotation_text=f"entry_z = {ENTRY_Z_SCORE}")
            fig_zt.add_hline(y=0, line_color="#2a3048")
            fig_zt.update_layout(
                title="Z-Score dos Resíduos ao Longo do Tempo (Top 5 Shorts)",
                **bl(360))
            uf(fig_zt)
            st.plotly_chart(fig_zt, use_container_width=True)

# ── Tab 4: Trades ─────────────────────────────────────────────────────────────
with tab4:
    tr_df = bt['trades']
    if len(tr_df) == 0:
        st.warning("Nenhum trade. Ajuste os parâmetros.")
    else:
        ca4, cb4 = st.columns(2)
        with ca4:
            fig_d = go.Figure(go.Histogram(
                x=tr_df['ret_short_pct'], nbinsx=40,
                marker_color='#58a6ff', opacity=.85))
            fig_d.add_vline(x=0, line_dash="dash", line_color="#f85149")
            mu = tr_df['ret_short_pct'].mean()
            fig_d.add_vline(x=mu, line_dash="dot", line_color="#3fb950",
                           annotation_text=f"μ={mu:.2f}%")
            fig_d.update_layout(
                title="Distribuição Retornos SHORT por Trade (%)", **bl(320))
            uf(fig_d)
            st.plotly_chart(fig_d, use_container_width=True)

            wr_t = (tr_df.groupby('ticker')
                         .apply(lambda x: (x['pnl']>0).mean()*100)
                         .sort_values())
            fig_wr = go.Figure(go.Bar(
                x=wr_t.values, y=wr_t.index, orientation='h',
                marker_color=['#3fb950' if v>=50 else '#f85149'
                              for v in wr_t.values]))
            fig_wr.add_vline(x=50, line_dash="dash", line_color="#8b949e")
            fig_wr.update_layout(title="Win Rate por Ação (%)", **bl(380))
            uf(fig_wr)
            st.plotly_chart(fig_wr, use_container_width=True)

        with cb4:
            pnl_t = (tr_df.groupby('ticker')['pnl']
                          .sum().sort_values())
            fig_pt = go.Figure(go.Bar(
                x=pnl_t.values, y=pnl_t.index, orientation='h',
                marker_color=['#3fb950' if v>=0 else '#f85149'
                              for v in pnl_t.values]))
            fig_pt.update_layout(title="P&L Total por Ação ($)", **bl(380))
            uf(fig_pt)
            st.plotly_chart(fig_pt, use_container_width=True)

            zr = (tr_df.groupby('ticker')
                        .agg(z_mean=('z_score','mean'),
                             ret_mean=('ret_short_pct','mean'))
                        .reset_index())
            fig_zr = go.Figure(go.Scatter(
                x=zr['z_mean'], y=zr['ret_mean'],
                mode='markers+text', text=zr['ticker'],
                textposition='top center',
                marker=dict(color='#58a6ff', size=10,
                            line=dict(color='#bc8cff',width=1))))
            fig_zr.add_vline(x=ENTRY_Z_SCORE, line_dash="dash",
                             line_color="#d29922")
            fig_zr.add_hline(y=0, line_color="#2a3048")
            fig_zr.update_layout(
                title="Z-Score Entrada vs Retorno Médio Short (%)",
                xaxis_title="Z-Score médio de entrada",
                yaxis_title="Retorno médio (%)", **bl(380))
            uf(fig_zr)
            st.plotly_chart(fig_zr, use_container_width=True)

        st.markdown("#### 📋 Histórico de Trades")
        st.dataframe(
            tr_df.sort_values('rebal_dt', ascending=False)
                 .style.format({
                     'z_score':'{:.4f}','weight_pct':'{:.2f}%',
                     'notional':'${:,.0f}','pe':'${:.2f}','px':'${:.2f}',
                     'ret_short_pct':'{:+.3f}%','pnl':'${:+,.2f}',
                 }).background_gradient(subset=['ret_short_pct'], cmap='RdYlGn'),
            use_container_width=True
        )

# ── Tab 5: Código LEAN ────────────────────────────────────────────────────────
with tab5:
    st.markdown("### 📚 Algoritmo Original — QuantConnect LEAN")
    st.markdown("O dashboard replica fielmente cada passo de `_compute_weights_by_symbol`:")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### LEAN (Python)")
        st.code("""
# 1. Log + dropna + centrar
sample = np.log(history.dropna(axis=1))
sample = sample - sample.mean()

# 2. PCA fit
pca = PCA().fit(sample)

# 3. Factors: primeiros N componentes
factors = np.dot(sample, pca.components_.T)[:, :N]

# 4. Add constant (intercepto OLS)
factors = sm.add_constant(factors)

# 5. OLS por symbol
ols = {s: sm.OLS(sample[s], factors).fit()
       for s in sample.columns}

# 6. Resíduos
residuals = pd.DataFrame(
    {s: m.resid for s, m in ols.items()},
    index=sample.index)

# 7. Z-score do último dia
z = ((residuals - residuals.mean())
     / residuals.std()).iloc[-1]

# 8. Seleção & peso
sel   = z[z < entry_z_score]
denom = sel.abs().sum()
weights = sel * (1 / denom)   # negativo → SHORT
""", language="python")

    with col_b:
        st.markdown("#### Streamlit (replica idêntica)")
        st.code("""
def compute_weights_by_symbol(history_close,
                               pca_components,
                               entry_z):
    sample = np.log(history_close.dropna(axis=1))
    sample = sample - sample.mean()

    pca_model = PCA().fit(sample)
    factors_raw = np.dot(
        sample,
        pca_model.components_.T
    )[:, :pca_components]

    factors = sm.add_constant(factors_raw)

    ols_models = {
        col: sm.OLS(sample[col], factors).fit()
        for col in sample.columns
    }
    residuals = pd.DataFrame(
        {col: m.resid
         for col, m in ols_models.items()},
        index=sample.index
    )
    z_scores = (
        (residuals - residuals.mean())
        / residuals.std()
    ).iloc[-1]

    selected = z_scores[z_scores < entry_z]
    denom    = selected.abs().sum()
    weights  = selected * (1.0 / denom)
    return weights.to_dict(), z_scores, ...
""", language="python")

    st.markdown("---")
    st.markdown("""
    #### Mapeamento LEAN → Streamlit

    | Conceito LEAN | Equivalente no Dashboard |
    |:--|:--|
    | `self.set_cash(100000)` | `CAPITAL` na sidebar |
    | `self._lookback_days = 60` | Slider `_lookback_days` |
    | `self._pca_components = 3` | Slider `_pca_components` |
    | `self._entry_z_score = -1.5` | Slider `_entry_z_score` |
    | `f.price > 5` | Slider `Filtro preço ($)` |
    | `dollar_volume[-20:]` | Top-N por dollar vol médio |
    | `self.date_rules.month_start("SPY")` | `resample('MS')` |
    | `free_portfolio_value_percentage = 0.05` | Slider `free_portfolio_value_percentage` |
    | `set_holdings(targets, True)` | Liquidação imediata do portfolio anterior |
    | `set_warm_up(365)` | `LOOKBACK_DAYS + buffer` de dados extras |
    """)

st.markdown("---")
with st.expander("🗂️ Preços brutos — últimas 30 barras"):
    st.dataframe(res['close'][univ].tail(30).style.format("${:.2f}"),
                 use_container_width=True)

st.caption("⚠️ Dashboard educacional. Não constitui recomendação de investimento.")
