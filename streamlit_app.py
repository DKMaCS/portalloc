# streamlit_app.py
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import cvxpy as cp
from pypfopt import risk_models, EfficientFrontier

st.set_page_config(page_title="Efficient Frontier & CML — True Constrained", layout="wide")
st.title("Efficient Frontier & CML (True Constrained)")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Inputs")
    rf = st.number_input("Risk-free rate (annual)", value=0.02, step=0.001, format="%.4f")
    gamma = st.number_input("Risk aversion γ (≥ 0)", value=300.0, step=10.0, min_value=0.0, format="%.2f")
    freq = st.selectbox("Periods per year", [252, 52, 12], index=0)
    allow_short = st.checkbox("Allow shorting (long/short)", value=False)
    lower_bound = -1.0 if allow_short else 0.0
    upper_bound = 1.0
    n_frontier = st.slider("Frontier points", 30, 300, 140)
    frontier_mode = st.selectbox("Frontier mode", ["Target return (recommended)", "Target volatility (σ)"])

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
Sigma = risk_models.sample_cov(rets, frequency=freq, returns_data=True)
mu_vec, Sigma_mat = mu.values, Sigma.values

# ---------------- Solvers ----------------
def solve_gamma_portfolio(mu_vec, Sigma_mat, rf, gamma, lb, ub):
    """max (μ−rf)^T w − ½γ w^T Σ w  s.t. ∑w=1, lb≤w≤ub"""
    n = len(mu_vec)
    w = cp.Variable(n)
    mu_ex = mu_vec - rf
    objective = cp.Maximize(mu_ex @ w - 0.5 * gamma * cp.quad_form(w, Sigma_mat))
    constraints = [cp.sum(w) == 1, w >= lb, w <= ub]
    prob = cp.Problem(objective, constraints)
    for solver in [cp.OSQP, cp.SCS, cp.ECOS]:
        try:
            prob.solve(solver=solver, verbose=False)
            if w.value is not None:
                return np.asarray(w.value, dtype=float).reshape(-1)
        except Exception:
            pass
    raise RuntimeError("CVXPY failed to solve γ-portfolio.")

def portfolio_stats(w, mu_vec, Sigma_mat, rf):
    ret = float(w @ mu_vec)
    vol = float(np.sqrt(max(w @ Sigma_mat @ w, 0.0)))
    sharpe = (ret - rf) / max(vol, 1e-12)
    return ret, vol, sharpe

def feasible_return_bounds(mu_vec, lb, ub):
    """Compute min/max achievable μ under sum(w)=1 and lb≤w≤ub."""
    n = len(mu_vec)
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1, w >= lb, w <= ub]
    pmax = cp.Problem(cp.Maximize(mu_vec @ w), cons)
    pmin = cp.Problem(cp.Minimize(mu_vec @ w), cons)
    for solver in [cp.OSQP, cp.SCS, cp.ECOS]:
        try:
            pmax.solve(solver=solver, verbose=False)
            pmin.solve(solver=solver, verbose=False)
            if pmax.status not in ("infeasible", "unbounded") and pmin.status not in ("infeasible", "unbounded"):
                return float(pmin.value), float(pmax.value)
        except Exception:
            continue
    raise RuntimeError("Could not bracket feasible return range.")

def compute_frontier_target_return_feasible(mu, Sigma, rf, lb, ub, n_points):
    """Sweep target return over the *feasible* range starting at GMV return,
    but back off the cap and dedupe near-identical points."""
    # GMV stats
    ef_gmv = EfficientFrontier(mu, Sigma, weight_bounds=(lb, ub))
    ef_gmv.min_volatility()
    gmv_ret, gmv_vol, _ = ef_gmv.portfolio_performance(risk_free_rate=rf)

    # Feasible μ-range under bounds (cap is μ_max for long-only)
    r_lo, r_hi = feasible_return_bounds(mu.values, lb, ub)
    r_cap = float(mu.max())  # max achievable μ under long-only
    r_hi = min(r_hi, r_cap) - 1e-8
    r_start = max(gmv_ret, r_lo)

    targets = np.linspace(r_start, r_hi, n_points)

    vols, rets = [], []
    last_w = None
    for r in targets:
        ef = EfficientFrontier(mu, Sigma, weight_bounds=(lb, ub))
        try:
            ef.efficient_return(target_return=r)
            # dedupe by weights (more robust than returns alone)
            w_dict = ef.clean_weights(cutoff=1e-6)
            w_vec = np.array([w_dict.get(t, 0.0) for t in mu.index])
            if last_w is not None and np.linalg.norm(w_vec - last_w, 1) <= 1e-6:
                continue
            ret, vol, _ = ef.portfolio_performance(risk_free_rate=rf)
            rets.append(ret); vols.append(vol)
            last_w = w_vec
        except Exception:
            continue
    return np.array(vols), np.array(rets)


def compute_frontier_target_vol(mu, Sigma, rf, lb, ub, n_points):
    """Sweep target volatility (efficient_risk) to get even σ spacing."""
    ef_gmv = EfficientFrontier(mu, Sigma, weight_bounds=(lb, ub))
    ef_gmv.min_volatility()
    gmv_ret, gmv_vol, _ = ef_gmv.portfolio_performance(risk_free_rate=rf)

    # crude high-σ anchor: highest-μ asset vol or 2×GMV σ (whichever larger)
    idx_hi = int(np.argmax(mu.values))
    hi_vol = float(np.sqrt(Sigma.values[idx_hi, idx_hi]))
    sig_targets = np.linspace(gmv_vol, max(hi_vol, 2*gmv_vol), n_points)

    vols, rets = [], []
    for s in sig_targets:
        ef = EfficientFrontier(mu, Sigma, weight_bounds=(lb, ub))
        try:
            ef.efficient_risk(target_volatility=s)
            ret, vol, _ = ef.portfolio_performance(risk_free_rate=rf)
            vols.append(vol); rets.append(ret)
        except Exception:
            continue
    return np.array(vols), np.array(rets)

def clean_clip_weights(w, lb=0.0, ub=1.0, cutoff=1e-6):
    w = np.array(w, float).ravel()
    w[np.abs(w) < cutoff] = 0.0
    w = np.clip(w, lb, ub)
    s = w.sum()
    return w if s == 0 else w / s

# ---------------- Key portfolios ----------------
# γ portfolio (investor preference)
w_gamma = solve_gamma_portfolio(mu_vec, Sigma_mat, rf, gamma, lower_bound, upper_bound)
w_gamma = clean_clip_weights(w_gamma, lb=lower_bound, ub=upper_bound, cutoff=1e-4)
gamma_ret, gamma_vol, gamma_sharpe = portfolio_stats(w_gamma, mu_vec, Sigma_mat, rf)


# Tangency (max Sharpe)
ef_tan = EfficientFrontier(mu, Sigma, weight_bounds=(lower_bound, upper_bound))
ef_tan.max_sharpe(risk_free_rate=rf)
w_tan_dict = ef_tan.clean_weights(cutoff=1e-6)   # <- important
w_tan = np.array([w_tan_dict.get(t, 0.0) for t in tickers])
tan_ret, tan_vol, tan_sharpe = ef_tan.portfolio_performance(risk_free_rate=rf)


# GMV (min variance)
ef_gmv = EfficientFrontier(mu, Sigma, weight_bounds=(lower_bound, upper_bound))
ef_gmv.min_volatility()
w_gmv_dict = ef_gmv.clean_weights(cutoff=1e-6)   # <- important
w_gmv = np.array([w_gmv_dict.get(t, 0.0) for t in tickers])
gmv_ret, gmv_vol, _ = ef_gmv.portfolio_performance(risk_free_rate=rf)


# Frontier curve (preference-free)
if frontier_mode.startswith("Target return"):
    ef_vols, ef_rets = compute_frontier_target_return_feasible(mu, Sigma, rf, lower_bound, upper_bound, n_frontier)
else:
    ef_vols, ef_rets = compute_frontier_target_vol(mu, Sigma, rf, lower_bound, upper_bound, n_frontier)

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(7.8, 5.6))

if ef_vols.size:
    order = np.argsort(ef_vols)
    ax.plot(ef_vols[order], ef_rets[order], lw=1.6, alpha=0.95, label="Efficient Frontier")

# key points
ax.scatter([gmv_vol], [gmv_ret], marker="D", label="GMV")
ax.scatter([tan_vol], [tan_ret], marker="*", s=140, label="Tangency")
ax.scatter([gamma_vol], [gamma_ret], marker="o", label=f"Γ-Portfolio (γ={gamma:.2f})")

# CML
xmax = max([tan_vol, ef_vols.max() if ef_vols.size else tan_vol]) * 1.05
x = np.linspace(0, xmax, 200)
cml = rf + (tan_ret - rf) * (x / max(tan_vol, 1e-12))
ax.plot(x, cml, linestyle="--", label="CML")

ax.set_xlim(0, xmax)
ax.margins(y=0.05)
ax.set_xlabel("Volatility (σ, annualized)")
ax.set_ylabel("Expected Return (μ, annualized)")
ax.set_title("Efficient Frontier — Preference-Free Curve + Investor Point (γ)")
ax.legend(loc="best")
st.pyplot(fig, clear_figure=True)

# ---------------- Tables ----------------
st.markdown("### Portfolio Performance Summary")
summary_df = pd.DataFrame({
    "Portfolio": ["GMV", "Tangency", f"Gamma (γ={gamma:.2f})"],
    "Expected Return": [gmv_ret, tan_ret, gamma_ret],
    "Volatility": [gmv_vol, tan_vol, gamma_vol],
    "Sharpe Ratio": [
        (gmv_ret - rf) / max(gmv_vol, 1e-12),
        (tan_ret - rf) / max(tan_vol, 1e-12),
        (gamma_ret - rf) / max(gamma_vol, 1e-12),
    ]
})
st.dataframe(summary_df.round(4), use_container_width=True)

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
    st.subheader(f"Risk-averse (γ={gamma:.2f})")
    st.dataframe(weights_df(w_gamma, tickers), use_container_width=True)

# ---------------- Export ----------------
export_df = pd.DataFrame({
    "Ticker": tickers,
    "GMV": np.array([w_gmv_dict.get(t, 0.0) for t in tickers]),
    "Tangency": np.array([w_tan_dict.get(t, 0.0) for t in tickers]),
    f"Gamma_{gamma:.2f}": w_gamma,                      # already cleaned
})

st.download_button(
    "Download Weights CSV",
    data=export_df.to_csv(index=False),
    file_name="weights_true_constrained.csv",
    mime="text/csv"
)

st.caption(
    "Frontier drawn over the *feasible* return range (or target σ), eliminating visual flattening. "
    "γ selects the investor-optimal point on that curve. μ, Σ from log returns; bounds match the sidebar."
)
