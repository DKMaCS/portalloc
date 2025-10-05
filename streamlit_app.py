# streamlit_app.py
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import cvxpy as cp
from pypfopt import risk_models

st.set_page_config(page_title="Efficient Frontier — γ sweep with leverage limit", layout="wide")
st.title("Efficient Frontier (∑w=1, ‖w‖₁ ≤ L) — γ sweep")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Inputs")
    rf = st.number_input("Risk-free rate (annual)", value=0.02, step=0.001, format="%.4f")   # for Sharpe/CML display only
    gamma_sel = st.number_input("Selected γ (risk aversion)", value=10.0, step=1.0, min_value=0.0, format="%.3f")
    freq = st.selectbox("Periods per year", [252, 52, 12], index=0)

    # Frontier resolution & γ range
    n_frontier = st.slider("Frontier points (γ sweep)", 30, 400, 160)
    g_min = st.number_input("γ min (log sweep)", value=1e-2, format="%.4g")
    g_max = st.number_input("γ max (log sweep)", value=1e3,  format="%.4g")

    # *** leverage (gross exposure) limit ***
    L1 = st.number_input("Leverage (‖w‖₁ ≤ L)", value=2.0, step=0.1, min_value=1.0, format="%.2f")

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
mu = rets.mean() * freq
# Use shrinkage to avoid ill-conditioning that exaggerates leverage effects
Sigma = risk_models.CovarianceShrinkage(rets, returns_data=True).ledoit_wolf()  # annualized already
mu_vec, Sigma_mat = mu.values, Sigma.values
n = len(mu_vec)

# ---------------- Helpers ----------------
def clean(w, cutoff=1e-10):
    w = np.array(w, float).ravel()
    w[np.abs(w) < cutoff] = 0.0
    s = w.sum()
    return w if s == 0 else w / s

def stats(w, rf_):
    r = float(w @ mu_vec)
    s2 = float(w @ Sigma_mat @ w)
    s = float(np.sqrt(max(s2, 0.0)))
    sh = (r - rf_) / max(s, 1e-12)
    return r, s, sh

# ---------------- γ-portfolio solver (same constraint as frontier) ----------------
# Theoretical mean–variance:  minimize 0.5 w^T Σ w − (1/γ) μ^T w
# subject to 1^T w = 1  and  ‖w‖₁ ≤ L  (leverage cap)
def solve_gamma(mu_vec, Sigma_mat, gamma, L1):
    gamma = float(max(gamma, 1e-12))
    w = cp.Variable(len(mu_vec))
    obj = 0.5 * cp.quad_form(w, Sigma_mat) - (1.0/gamma) * (mu_vec @ w)
    cons = [cp.sum(w) == 1, cp.norm1(w) <= L1]
    prob = cp.Problem(cp.Minimize(obj), cons)
    for solver in (cp.OSQP, cp.SCS, cp.ECOS):
        try:
            prob.solve(solver=solver, verbose=False)
            if w.value is not None:
                return np.asarray(w.value, float).ravel()
        except Exception:
            pass
    raise RuntimeError("γ solve failed")

# ---------------- Trace the EF by sweeping γ (same feasible set) ----------------
gammas = np.logspace(np.log10(g_max), np.log10(g_min), n_frontier)  # high→low risk aversion
ef_rets, ef_vols, ef_ws = [], [], []
last_w = None
for g in gammas:
    w = clean(solve_gamma(mu_vec, Sigma_mat, g, L1))
    if last_w is not None and np.linalg.norm(w - last_w, 1) <= 1e-8:
        continue
    r, s, _ = stats(w, rf)
    ef_rets.append(r); ef_vols.append(s); ef_ws.append(w); last_w = w
ef_rets, ef_vols = np.array(ef_rets), np.array(ef_vols)

# ---------------- Your selected γ portfolio (lies on that curve) ----------------
w_gamma = clean(solve_gamma(mu_vec, Sigma_mat, gamma_sel, L1))
gamma_ret, gamma_vol, gamma_sharpe = stats(w_gamma, rf)

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(5.0, 3.6), dpi=120)

order = np.argsort(ef_vols)
ax.plot(ef_vols[order], ef_rets[order], lw=1.8, label=f"Efficient Frontier (γ-sweep, ‖w‖₁ ≤ {L1:g})")

ax.scatter([gamma_vol], [gamma_ret], label=f"Γ-Portfolio (γ={gamma_sel:.2f})", s=40)

# Optional: CML through the *constrained* tangency (estimate by slope at γ near risk-seeking end)
# (We show it lightly to avoid confusion.)
if len(ef_vols) >= 2:
    x0, y0 = ef_vols[-2], ef_rets[-2]
    x1, y1 = ef_vols[-1], ef_rets[-1]
    slope = (y1 - y0) / max(x1 - x0, 1e-12)
    xmax = 1.05 * max(gamma_vol, ef_vols.max())
    x = np.linspace(0, xmax, 200)
    cml = rf + slope * x
    ax.plot(x, cml, linestyle="--", alpha=0.35, label="Ref. line")

ax.set_xlim(0, 1.05 * max(ef_vols.max(), gamma_vol))
ax.margins(y=0.05)
ax.set_xlabel("Volatility (σ, annualized)")
ax.set_ylabel("Expected Return (μ, annualized)")
ax.set_title("Efficient Frontier with Leverage Cap (∑w=1, ‖w‖₁ ≤ L)")
ax.legend(loc="best", fontsize=8)
st.pyplot(fig, clear_figure=True, use_container_width=False)

# ---------------- Tables ----------------
st.markdown("### Γ-Portfolio Performance")
summary_df = pd.DataFrame({
    "Portfolio": [f"Gamma (γ={gamma_sel:.2f})"],
    "Expected Return": [gamma_ret],
    "Volatility": [gamma_vol],
    "Sharpe Ratio": [(gamma_ret - rf) / max(gamma_vol, 1e-12)],
})
st.dataframe(summary_df.round(6), use_container_width=True)

st.markdown("### Γ-Portfolio Weights")
def weights_df(weights, tickers):
    df = pd.DataFrame({"Ticker": tickers, "Weight": weights})
    df["Weight"] = df["Weight"].round(6)
    return df[df["Weight"].abs() > 1e-8].sort_values("Weight", ascending=False).reset_index(drop=True)
st.dataframe(weights_df(w_gamma, tickers), use_container_width=True)

# ---------------- Export ----------------
export_df = pd.DataFrame({"Ticker": tickers, f"Gamma_{gamma_sel:.2f}": w_gamma})
st.download_button(
    "Download Weights CSV",
    data=export_df.to_csv(index=False),
    file_name="weights_gamma_frontier_L1.csv",
    mime="text/csv"
)

st.caption(
    "We add a leverage cap ‖w‖₁ ≤ L so γ-portfolios stay in a realistic region. "
    "The frontier is traced by the same γ optimization, so the Γ point lies on the curve. "
    "Increase L to allow more leverage; decrease it to keep σ, μ in range and reveal curvature."
)
