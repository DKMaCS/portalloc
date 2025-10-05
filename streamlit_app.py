# streamlit_app.py
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import cvxpy as cp

st.set_page_config(page_title="Efficient Frontier — γ sweep (constrained CML)",
                   layout="wide")
st.title("Efficient Frontier (∑w=1, ‖w‖₁ ≤ L) — γ sweep with constrained CML")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Inputs")
    rf = st.number_input("Risk-free rate (annual, for Sharpe/CML only)",
                         value=0.02, step=0.001, format="%.4f")
    gamma_sel = st.number_input("Selected γ (risk aversion)",
                                value=10.0, step=1.0, min_value=0.0, format="%.3f")
    freq = st.selectbox("Periods per year", [252, 52, 12], index=0)

    n_frontier = st.slider("Frontier points (γ sweep)", 30, 400, 160)
    g_min = st.number_input("γ min (log sweep)", value=1e-2, format="%.4g")
    g_max = st.number_input("γ max (log sweep)", value=1e3,  format="%.4g")

    L1 = st.number_input("Leverage cap  ‖w‖₁ ≤ L", value=2.0,
                         step=0.1, min_value=1.0, format="%.2f")

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

# ---------------- μ and Σ from log returns (annualized) ----------------
rets = np.log(prices).diff().dropna()
mu = rets.mean() * freq             # annualized expected returns
Sigma = rets.cov() * freq           # annualized covariance (no sklearn dependency)

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

# ---------------- Solvers (CVXPy) ----------------
# Mean–variance utility with leverage cap:
# minimize 0.5 w^T Σ w − (1/γ) μ^T w
# subject to 1^T w = 1,  ‖w‖₁ ≤ L1
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

# Global minimum-variance under same constraints
def solve_gmv(Sigma_mat, L1):
    w = cp.Variable(Sigma_mat.shape[0])
    obj = cp.quad_form(w, Sigma_mat)
    cons = [cp.sum(w) == 1, cp.norm1(w) <= L1]
    prob = cp.Problem(cp.Minimize(obj), cons)
    for solver in (cp.OSQP, cp.SCS, cp.ECOS):
        try:
            prob.solve(solver=solver, verbose=False)
            if w.value is not None:
                return np.asarray(w.value, float).ravel()
        except Exception:
            pass
    raise RuntimeError("GMV solve failed")

# ---------------- Trace EF by sweeping γ (same feasible set) ----------------
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

# ---------------- Selected γ portfolio (lies on that curve) ----------------
w_gamma = clean(solve_gamma(mu_vec, Sigma_mat, gamma_sel, L1))
gamma_ret, gamma_vol, gamma_sharpe = stats(w_gamma, rf)

# ---------------- GMV & Constrained Tangency (from EF itself) ----------------
w_gmv = clean(solve_gmv(Sigma_mat, L1))
gmv_ret, gmv_vol, _ = stats(w_gmv, rf)

if ef_vols.size == 0:
    st.error("Frontier empty; check inputs.")
    st.stop()

sharpe_arr = (ef_rets - rf) / np.maximum(ef_vols, 1e-12)
i_tan = int(np.nanargmax(sharpe_arr))
tan_vol = float(ef_vols[i_tan])
tan_ret = float(ef_rets[i_tan])
w_tan = ef_ws[i_tan]

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(5.0, 3.6), dpi=120)

order = np.argsort(ef_vols)
ax.plot(ef_vols[order], ef_rets[order], lw=1.8,
        label=f"Efficient Frontier (γ sweep, ‖w‖₁ ≤ {L1:g})")

ax.scatter([gmv_vol], [gmv_ret], marker="D", label="GMV")
ax.scatter([tan_vol], [tan_ret], marker="*", s=110, label="Tangency (constrained)")
ax.scatter([gamma_vol], [gamma_ret], marker="o", label=f"Γ-Portfolio (γ={gamma_sel:.2f})")

# Constrained CML: line from rf through constrained tangency
xmax = 1.05 * max(tan_vol, gamma_vol, ef_vols.max())
x = np.linspace(0, xmax, 200)
slope = (tan_ret - rf) / max(tan_vol, 1e-12)
ax.plot(x, rf + slope * x, linestyle="--", label="CML (constrained)")

ax.set_xlim(0, xmax)
ax.margins(y=0.05)
ax.set_xlabel("Volatility (σ, annualized)")
ax.set_ylabel("Expected Return (μ, annualized)")
ax.set_title("Efficient Frontier with Leverage Cap (∑w=1, ‖w‖₁ ≤ L) + Constrained CML")
ax.legend(loc="best", fontsize=8)
st.pyplot(fig, clear_figure=True, use_container_width=False)

# ---------------- Tables ----------------
st.markdown("### Portfolio Performance Summary")
summary_df = pd.DataFrame({
    "Portfolio": [f"Gamma (γ={gamma_sel:.2f})", "GMV", "Tangency (constrained)"],
    "Expected Return": [gamma_ret, gmv_ret, tan_ret],
    "Volatility": [gamma_vol, gmv_vol, tan_vol],
    "Sharpe Ratio": [
        (gamma_ret - rf) / max(gamma_vol, 1e-12),
        (gmv_ret - rf) / max(gmv_vol, 1e-12),
        (tan_ret - rf) / max(tan_vol, 1e-12),
    ]
})
st.dataframe(summary_df.round(6), use_container_width=True)

st.markdown("### Weights")
def weights_df(weights, tickers):
    df = pd.DataFrame({"Ticker": tickers, "Weight": weights})
    df["Weight"] = df["Weight"].round(6)
    return df[df["Weight"].abs() > 1e-8].sort_values("Weight", ascending=False).reset_index(drop=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader(f"Γ-Portfolio (γ={gamma_sel:.2f})")
    st.dataframe(weights_df(w_gamma, tickers), use_container_width=True)
with c2:
    st.subheader("GMV")
    st.dataframe(weights_df(w_gmv, tickers), use_container_width=True)
with c3:
    st.subheader("Tangency (constrained)")
    st.dataframe(weights_df(w_tan, tickers), use_container_width=True)

# ---------------- Export ----------------
export_df = pd.DataFrame({
    "Ticker": tickers,
    f"Gamma_{gamma_sel:.2f}": w_gamma,
    "GMV": w_gmv,
    "Tangency_constrained": w_tan,
})
st.download_button(
    "Download Weights CSV",
    data=export_df.to_csv(index=False),
    file_name="weights_gamma_frontier_constrained.csv",
    mime="text/csv"
)

st.caption(
    "Frontier and Γ-portfolio use the same γ optimization and leverage cap; "
    "the CML is drawn through the **constrained** tangency, so it touches the EF correctly. "
    "Increase L to allow more leverage (curve approaches a ray); decrease L for more curvature."
)
