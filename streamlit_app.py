import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import cvxpy as cp
from pypfopt import risk_models

st.set_page_config(page_title="Efficient Frontier & CML — Box Bounds", layout="wide")
st.title("Efficient Frontier & CML (Box Bounds; ∑w = 1)")

#  Fixed γ sweep (hard-coded)
G_SWEEP_MIN = 1e-2     # small γ => risk-seeking end
G_SWEEP_MAX = 1e4      # large γ => risk-averse end

#  Sidebar
with st.sidebar:
    st.header("Inputs")
    rf = st.number_input("Risk-free rate (annual)", value=0.02, step=0.001, format="%.4f")
    gamma = st.number_input("Risk aversion γ (≥ 0)", value=300.0, step=10.0, min_value=0.0, format="%.2f")
    freq = st.selectbox("Periods per year", [252, 52, 12], index=0)

    allow_short = st.checkbox("Allow shorting", value=False)
    lower_bound = -1.0 if allow_short else 0.0
    upper_bound = 1.0

    n_frontier = st.slider("Frontier points (γ sweep)", 30, 300, 140)

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

#  μ and Σ from log returns (annualized)
rets = np.log(prices).diff().dropna()
mu = np.exp(rets.mean() * freq) - 1.0
Sigma = risk_models.sample_cov(rets, frequency=freq, returns_data=True)  # annualized
mu_vec, Sigma_mat = mu.values, Sigma.values

#  Helpers
def clean_and_project(w, lb, ub, cutoff=1e-8):
    """Zero tiny weights, clip to [lb, ub], then renormalize to sum to 1."""
    w = np.array(w, float).ravel()
    w[np.abs(w) < cutoff] = 0.0
    w = np.clip(w, lb, ub)
    s = w.sum()
    return w if s == 0 else w / s

def portfolio_stats(w, mu_vec, Sigma_mat, rf):
    ret = float(w @ mu_vec)
    vol = float(np.sqrt(max(w @ Sigma_mat @ w, 0.0)))
    sharpe = (ret - rf) / max(vol, 1e-12)
    return ret, vol, sharpe

#  Solvers under BOX BOUNDS
def solve_gamma_portfolio(mu_vec, Sigma_mat, rf, gamma, lb, ub):
    """
    max (μ−rf)^T w − ½γ w^T Σ w
    s.t. 1^T w = 1, lb ≤ w ≤ ub
    """
    n = len(mu_vec)
    w = cp.Variable(n)
    mu_ex = mu_vec - rf
    objective = cp.Maximize(mu_ex @ w - 0.5 * gamma * cp.quad_form(w, Sigma_mat))
    cons = [cp.sum(w) == 1, w >= lb, w <= ub]
    prob = cp.Problem(objective, cons)
    for solver in (cp.OSQP, cp.SCS, cp.ECOS):
        try:
            prob.solve(solver=solver, verbose=False)
            if w.value is not None:
                return np.asarray(w.value, float).ravel()
        except Exception:
            pass
    raise RuntimeError("CVXPY failed in γ solve.")

def solve_gmv(Sigma_mat, lb, ub):
    """Min variance under 1^T w = 1, lb ≤ w ≤ ub."""
    n = Sigma_mat.shape[0]
    w = cp.Variable(n)
    obj = cp.quad_form(w, Sigma_mat)
    cons = [cp.sum(w) == 1, w >= lb, w <= ub]
    prob = cp.Problem(cp.Minimize(obj), cons)
    for solver in (cp.OSQP, cp.SCS, cp.ECOS):
        try:
            prob.solve(solver=solver, verbose=False)
            if w.value is not None:
                return np.asarray(w.value, float).ravel()
        except Exception:
            pass
    raise RuntimeError("CVXPY failed in GMV solve.")

#  Frontier traced by γ (same constraints)
def compute_frontier_by_gamma(mu_vec, Sigma_mat, rf, lb, ub, n_points):
    gammas = np.logspace(np.log10(G_SWEEP_MAX), np.log10(G_SWEEP_MIN), n_points)  # high→low risk aversion
    vols, rets, ws = [], [], []
    last_w = None
    for g in gammas:
        w = solve_gamma_portfolio(mu_vec, Sigma_mat, rf, g, lb, ub)
        w = clean_and_project(w, lb, ub, cutoff=1e-8)
        # dedupe near-identical corners
        if last_w is not None and np.linalg.norm(w - last_w, 1) <= 1e-8:
            continue
        r, s, _ = portfolio_stats(w, mu_vec, Sigma_mat, rf)
        rets.append(r); vols.append(s); ws.append(w); last_w = w
    vols = np.array(vols); rets = np.array(rets)
    order = np.argsort(vols)
    return vols[order], rets[order], [ws[i] for i in order]

#  Key portfolios
# γ portfolio (investor choice) — BOX BOUNDS
w_gamma = solve_gamma_portfolio(mu_vec, Sigma_mat, rf, gamma, lower_bound, upper_bound)
w_gamma = clean_and_project(w_gamma, lower_bound, upper_bound, cutoff=1e-8)
gamma_ret, gamma_vol, gamma_sharpe = portfolio_stats(w_gamma, mu_vec, Sigma_mat, rf)

# GMV (min variance) — BOX BOUNDS
w_gmv = solve_gmv(Sigma_mat, lower_bound, upper_bound)
w_gmv = clean_and_project(w_gmv, lower_bound, upper_bound, cutoff=1e-8)
gmv_ret, gmv_vol, _ = portfolio_stats(w_gmv, mu_vec, Sigma_mat, rf)

# Frontier (preference-free) — traced by γ with SAME constraints (fixed sweep)
ef_vols, ef_rets, ef_ws = compute_frontier_by_gamma(
    mu_vec, Sigma_mat, rf, lower_bound, upper_bound, n_frontier
)

# Constrained tangency taken FROM the same EF (consistent CML)
if ef_vols.size:
    sharpe_arr = (ef_rets - rf) / np.maximum(ef_vols, 1e-12)
    i_tan = int(np.nanargmax(sharpe_arr))
    tan_vol = float(ef_vols[i_tan])
    tan_ret = float(ef_rets[i_tan])
    w_tan = ef_ws[i_tan]
else:
    tan_vol = tan_ret = 0.0
    w_tan = np.zeros_like(mu_vec)

#  Plot
fig, ax = plt.subplots(figsize=(4, 3), dpi=120)

if ef_vols.size:
    ax.plot(ef_vols, ef_rets, lw=1.8, alpha=0.95, label="Efficient Frontier (γ sweep, box bounds)")

# key points
ax.scatter([gmv_vol], [gmv_ret], marker="D", label="GMV")
ax.scatter([tan_vol], [tan_ret], marker="*", s=120, label="Tangency (constrained)")
ax.scatter([gamma_vol], [gamma_ret], marker="o", label=f"Γ-Portfolio (γ={gamma:.2f})")

# CML through the constrained tangency
xmax = 1.05 * max(tan_vol, gamma_vol, (ef_vols.max() if ef_vols.size else 0.0))
x = np.linspace(0, xmax, 200)
slope = (tan_ret - rf) / max(tan_vol, 1e-12) if tan_vol > 0 else 0.0
ax.plot(x, rf + slope * x, linestyle="--", label="CML (constrained)")

ax.set_xlim(0, xmax)
ax.margins(y=0.05)
ax.set_xlabel("Volatility (σ, annualized)")
ax.set_ylabel("Expected Return (μ, annualized)")
ax.set_title(f"Efficient Frontier — Box Bounds [{lower_bound:.1f}, {upper_bound:.1f}] & Γ-Portfolio")
ax.legend(loc="best", fontsize=8)
st.pyplot(fig, clear_figure=True)

#  Tables
st.markdown("### Portfolio Performance Summary")
summary_df = pd.DataFrame({
    "Portfolio": ["GMV", "Tangency (constr.)", f"Gamma (γ={gamma:.2f})"],
    "Expected Return": [gmv_ret, tan_ret, gamma_ret],
    "Volatility": [gmv_vol, tan_vol, gamma_vol],
    "Sharpe Ratio": [
        (gmv_ret - rf)/max(gmv_vol, 1e-12),
        (tan_ret - rf)/max(tan_vol, 1e-12),
        (gamma_ret - rf)/max(gamma_vol, 1e-12),
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
    st.subheader("GMV")
    st.dataframe(weights_df(w_gmv, tickers), use_container_width=True)
with c2:
    st.subheader("Tangency (constr.)")
    st.dataframe(weights_df(w_tan, tickers), use_container_width=True)
with c3:
    st.subheader(f"Γ-Portfolio (γ={gamma:.2f})")
    st.dataframe(weights_df(w_gamma, tickers), use_container_width=True)

#  Export
export_df = pd.DataFrame({
    "Ticker": tickers,
    "GMV": w_gmv,
    "Tangency_constrained": w_tan,
    f"Gamma_{gamma:.2f}": w_gamma,
})
st.download_button(
    "Download Weights CSV",
    data=export_df.to_csv(index=False),
    file_name="weights_box_bounds_gamma_frontier.csv",
    mime="text/csv"
)

st.caption(
    "Frontier is generated by sweeping γ with fixed log-range (hard-coded) under the same box bounds. "
    "Kinks appear where bounds become active or the active asset set changes — expected with box constraints."
)
