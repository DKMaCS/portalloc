# streamlit_app.py
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pypfopt import risk_models, EfficientFrontier

st.set_page_config(page_title="Efficient Frontier & CML — Theoretical (γ-sweep)", layout="wide")
st.title("Efficient Frontier — Theoretical (Unconstrained, ∑w=1) via γ-sweep")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Inputs")
    rf = st.number_input("Risk-free rate (annual)", value=0.02, step=0.001, format="%.4f")
    gamma_sel = st.number_input("Risk aversion γ (≥ 0)", value=300.0, step=10.0, min_value=0.0, format="%.2f")
    freq = st.selectbox("Periods per year", [252, 52, 12], index=0)
    n_frontier = st.slider("Frontier points (γ sweep)", 30, 400, 160)
    g_min = st.number_input("γ min (log sweep)", value=1e-2, format="%.4g")
    g_max = st.number_input("γ max (log sweep)", value=1e4,  format="%.4g")

    uploaded = st.file_uploader(
        "Upload CSV of prices (first col: Date; other cols: tickers).",
        type=["csv"],
        help="Dates must parse; we'll ffill/bfill then use log returns."
    )

@st.cache_data(show_spinner=False)
def load_prices(file) -> pd.DataFrame:
    df = pd.read_csv(file, parse_dates=["Date"]).set_index("Date").sort_index()
    df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)
    return df.ffill().bfill()

if uploaded is None:
    st.info("Upload a CSV to continue.")
    st.stop()

prices = load_prices(uploaded)
if prices.shape[1] < 2:
    st.error("Need at least two assets.")
    st.stop()

tickers = list(prices.columns)

# ---------------- μ and Σ from log returns ----------------
rets = np.log(prices).diff().dropna()
mu = rets.mean() * freq                              # annualized μ
Sigma = risk_models.sample_cov(rets, frequency=freq, returns_data=True)  # annualized Σ
mu_vec, Sigma_mat = mu.values, Sigma.values
n = len(mu_vec)

# -------------- Closed-form γ-portfolio (KKT) ----------------
# Maximize (μ-rf)^T w − ½ γ w^T Σ w s.t. 1^T w = 1
# KKT → γ Σ w = (μ-rf) − λ 1  → w = (1/γ) Σ^{-1}[(μ-rf) − λ 1]
# with λ chosen so 1^T w = 1.
def gamma_weights_closed_form(mu_vec, Sigma_mat, rf, gamma, ridge=0.0):
    # tiny ridge for numerical stability if needed
    if ridge > 0:
        Sigma_mat = Sigma_mat + ridge * np.trace(Sigma_mat)/len(Sigma_mat) * np.eye(len(Sigma_mat))

    # Solve Σ x = b by linear solve (no explicit inverse)
    ones = np.ones_like(mu_vec)
    # u = Σ^{-1}(μ - rf*1), v = Σ^{-1}1
    u = np.linalg.solve(Sigma_mat, mu_vec - rf*ones)
    v = np.linalg.solve(Sigma_mat, ones)
    A = ones @ v               # = 1^T Σ^{-1} 1
    # Enforce budget 1^T w = 1:
    # w = (1/γ) u - (( (1/γ) * 1^T u - 1 )/A) v
    oneTu = ones @ u
    w = (1.0/gamma) * u - (( (1.0/gamma) * oneTu - 1.0 )/A) * v
    return w

def clean(w, cutoff=1e-10):
    w = np.array(w, float).ravel()
    w[np.abs(w) < cutoff] = 0.0
    s = w.sum()
    return w if s == 0 else w/s

def stats(w):
    r = float(w @ mu_vec)
    s2 = float(w @ Sigma_mat @ w)
    s = float(np.sqrt(max(s2, 0.0)))
    sh = (r - rf)/max(s, 1e-12)
    return r, s, sh

# -------------- Trace the theoretical frontier by γ-sweep --------------
gammas = np.logspace(np.log10(g_max), np.log10(g_min), n_frontier)  # high→low risk aversion
ef_rets, ef_vols, ef_ws = [], [], []
last_w = None
for g in gammas:
    w = gamma_weights_closed_form(mu_vec, Sigma_mat, rf, g)
    w = clean(w)
    if last_w is not None and np.linalg.norm(w-last_w, 1) <= 1e-10:
        continue
    r, s, _ = stats(w)
    ef_rets.append(r); ef_vols.append(s); ef_ws.append(w); last_w = w
ef_rets, ef_vols = np.array(ef_rets), np.array(ef_vols)

# -------------- Your selected γ portfolio (sits on that curve) --------------
# Guard against gamma=0
gamma_eff = max(float(gamma_sel), 1e-12)
w_gamma = clean(gamma_weights_closed_form(mu_vec, Sigma_mat, rf, gamma_eff))
gamma_ret, gamma_vol, gamma_sharpe = stats(w_gamma)

# -------------- GMV and Tangency (unconstrained) for reference --------------
# GMV is γ→∞ limit; Tangency is max Sharpe
ef_gmv = EfficientFrontier(mu, Sigma, weight_bounds=(None, None))
ef_gmv.min_volatility()
w_gmv = np.array([ef_gmv.clean_weights(1e-12).get(t,0.0) for t in tickers]); w_gmv = clean(w_gmv)
gmv_ret, gmv_vol, _ = ef_gmv.portfolio_performance(risk_free_rate=rf)

ef_tan = EfficientFrontier(mu, Sigma, weight_bounds=(None, None))
ef_tan.max_sharpe(risk_free_rate=rf)
w_tan = np.array([ef_tan.clean_weights(1e-12).get(t,0.0) for t in tickers]); w_tan = clean(w_tan)
tan_ret, tan_vol, tan_sharpe = ef_tan.portfolio_performance(risk_free_rate=rf)

# -------------- Plot --------------
fig, ax = plt.subplots(figsize=(5.0, 3.6), dpi=120)

order = np.argsort(ef_vols)
ax.plot(ef_vols[order], ef_rets[order], lw=1.8, label="Efficient Frontier (γ-sweep)")

ax.scatter([gmv_vol], [gmv_ret], marker="D", label="GMV")
ax.scatter([tan_vol], [tan_ret], marker="*", s=110, label="Tangency")
ax.scatter([gamma_vol], [gamma_ret], marker="o", label=f"Γ-Portfolio (γ={gamma_sel:.2f})")

# CML (reference)
xmax = 1.05 * max(tan_vol, gamma_vol, (ef_vols.max() if ef_vols.size else 0.0))
x = np.linspace(0, xmax, 200)
cml = rf + (tan_ret - rf) * (x / max(tan_vol, 1e-12))
ax.plot(x, cml, linestyle="--", label="CML")

ax.set_xlim(0, xmax)
ax.margins(y=0.05)
ax.set_xlabel("Volatility (σ, annualized)")
ax.set_ylabel("Expected Return (μ, annualized)")
ax.set_title("Theoretical Efficient Frontier (unconstrained) — traced by γ")
ax.legend(loc="best", fontsize=8)
st.pyplot(fig, clear_figure=True)

# -------------- Tables --------------
st.markdown("### Portfolio Performance Summary")
summary_df = pd.DataFrame({
    "Portfolio": ["GMV", "Tangency", f"Gamma (γ={gamma_sel:.2f})"],
    "Expected Return": [gmv_ret, tan_ret, gamma_ret],
    "Volatility": [gmv_vol, tan_vol, gamma_vol],
    "Sharpe Ratio": [
        (gmv_ret - rf)/max(gmv_vol,1e-12),
        (tan_ret - rf)/max(tan_vol,1e-12),
        (gamma_ret - rf)/max(gamma_vol,1e-12),
    ]
})
st.dataframe(summary_df.round(6), use_container_width=True)

def weights_df(weights, tickers):
    df = pd.DataFrame({"Ticker": tickers, "Weight": weights})
    df["Weight"] = df["Weight"].round(6)
    return df[df["Weight"].abs() > 1e-8].sort_values("Weight", ascending=False).reset_index(drop=True)

st.markdown("### Portfolio Weights")
c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("GMV")
    st.dataframe(weights_df(w_gmv, tickers), use_container_width=True)
with c2:
    st.subheader("Tangency")
    st.dataframe(weights_df(w_tan, tickers), use_container_width=True)
with c3:
    st.subheader(f"Γ-Portfolio (γ={gamma_sel:.2f})")
    st.dataframe(weights_df(w_gamma, tickers), use_container_width=True)

# -------------- Export --------------
export_df = pd.DataFrame({
    "Ticker": tickers,
    "GMV": w_gmv,
    "Tangency": w_tan,
    f"Gamma_{gamma_sel:.2f}": w_gamma,
})
st.download_button(
    "Download Weights CSV",
    data=export_df.to_csv(index=False),
    file_name="weights_unconstrained_gamma_frontier.csv",
    mime="text/csv"
)

st.caption(
    "Frontier is generated by the same γ-objective as the selected portfolio (unconstrained except ∑w=1). "
    "Therefore the Γ-portfolio lies on the plotted frontier to numerical precision."
)
