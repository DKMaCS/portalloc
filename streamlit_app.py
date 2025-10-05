import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import cvxpy as cp
from pypfopt import risk_models, EfficientFrontier

st.set_page_config(page_title="Efficient Frontier & CML — True Constrained", layout="wide")
st.title("Efficient Frontier & CML (True Constrained)")

# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.header("Inputs")
    rf = st.number_input("Risk-free rate (annual)", value=0.02, step=0.001, format="%.4f")
    gamma = st.number_input("Risk aversion γ (≥ 0)", value=300.0, step=10.0, min_value=0.0, format="%.2f")
    freq = st.selectbox("Periods per year", [252, 52, 12], index=0)
    allow_short = st.checkbox("Allow shorting (long/short)", value=False)
    n_frontier = st.slider("Frontier points", 30, 300, 120)
    frontier_mode = st.selectbox("Frontier mode", ["Target return (recommended)", "Gamma sweep (advanced)"])
    lower_bound = -1.0 if allow_short else 0.0
    upper_bound = 1.0

    uploaded = st.file_uploader(
        "Upload CSV of prices (first col: Date; other cols: tickers).",
        type=["csv"],
        help="Dates must parse; data are forward/back-filled, then log returns are used."
    )

@st.cache_data(show_spinner=False)
def load_prices(file) -> pd.DataFrame:
    df = pd.read_csv(file, parse_dates=["Date"]).set_index("Date").sort_index()
    df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)
    # match app: fill then returns
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

def compute_frontier_target_return(mu, Sigma, rf, lb, ub, n_points):
    mu_min, mu_max = float(mu.min()), float(mu.max())
    targets = np.linspace(mu_min, mu_max, n_points)
    ef_vols, ef_rets = [], []
    for r in targets:
        ef = EfficientFrontier(mu, Sigma, weight_bounds=(lb, ub))
        try:
            ef.efficient_return(target_return=r)
            ret, vol, _ = ef.portfolio_performance(risk_free_rate=rf)
            ef_vols.append(vol); ef_rets.append(ret)
        except Exception:
            # infeasible r given bounds; skip
            continue
    return np.array(ef_vols), np.array(ef_rets)

def compute_frontier_gamma(mu_vec, Sigma_mat, rf, lb, ub, n_points):
    # denser near small γ to sample high-return side better
    gammas = np.r_[np.linspace(0.01, 5, int(0.7*n_points)), np.logspace(np.log10(5), 3, int(0.3*n_points))]
    ef_vols, ef_rets = [], []
    for g in gammas:
        try:
            w = solve_gamma_portfolio(mu_vec, Sigma_mat, rf, g, lb, ub)
            r, v, _ = portfolio_stats(w, mu_vec, Sigma_mat, rf)
            ef_vols.append(v); ef_rets.append(r)
        except Exception:
            continue
    return np.array(ef_vols), np.array(ef_rets)

# ---------------- Key portfolios ----------------
mu_vec, Sigma_mat = mu.values, Sigma.values

# γ-portfolio (investor preference)
w_gamma = solve_gamma_portfolio(mu_vec, Sigma_mat, rf, gamma, lower_bound, upper_bound)
gamma_ret, gamma_vol, gamma_sharpe = portfolio_stats(w_gamma, mu_vec, Sigma_mat, rf)

# Tangency (max Sharpe)
ef_tan = EfficientFrontier(mu, Sigma, weight_bounds=(lower_bound, upper_bound))
w_tan_dict = ef_tan.max_sharpe(risk_free_rate=rf)
w_tan = np.array([w_tan_dict.get(t, 0.0) for t in tickers])
tan_ret, tan_vol, tan_sharpe = ef_tan.portfolio_performance(risk_free_rate=rf)

# GMV (min variance)
ef_gmv = EfficientFrontier(mu, Sigma, weight_bounds=(lower_bound, upper_bound))
w_gmv_dict = ef_gmv.min_volatility()
w_gmv = np.array([w_gmv_dict.get(t, 0.0) for t in tickers])
gmv_ret, gmv_vol, _ = ef_gmv.portfolio_performance(risk_free_rate=rf)

# Frontier points (choose mode)
if frontier_mode.startswith("Target"):
    ef_vols, ef_rets = compute_frontier_target_return(mu, Sigma, rf, lower_bound, upper_bound, n_frontier)
    frontier_label = "Efficient Frontier (target-return sweep)"
else:
    ef_vols, ef_rets = compute_frontier_gamma(mu_vec, Sigma_mat, rf, lower_bound, upper_bound, n_frontier)
    frontier_label = "Efficient Frontier (γ-sweep)"

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(7.5, 5.5))

# connect curve for readability
if ef_vols.size:
    order = np.argsort(ef_vols)
    ax.plot(ef_vols[order], ef_rets[order], lw=1.4, alpha=0.95, label=frontier_label)

# key points
ax.scatter([gmv_vol], [gmv_ret], marker="D", label="GMV")
ax.scatter([tan_vol], [tan_ret], marker="*", s=140, label="Tangency")
ax.scatter([gamma_vol], [gamma_ret], marker="o", label=f"Γ-Portfolio (γ={gamma:.2f})")

# CML
xmax = max([tan_vol, ef_vols.max() if ef_vols.size else tan_vol]) * 1.1
x = np.linspace(0, xmax, 200)
cml = rf + (tan_ret - rf) * (x / max(tan_vol, 1e-12))
ax.plot(x, cml, linestyle="--", label="CML")

ax.set_xlabel("Volatility (σ, annualized)")
ax.set_ylabel("Expected Return (μ, annualized)")
ax.set_title("Efficient Frontier — Market Opportunity Set + Investor Point (γ)")
ax.legend(loc="best")
st.pyplot(fig, clear_figure=True)

# ---------------- Performance summary ----------------
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

# ---------------- Weights tables ----------------
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
    st.subheader(f"Γ-Portfolio (γ={gamma:.2f})")
    st.dataframe(weights_df(w_gamma, tickers), use_container_width=True)

# ---------------- Export ----------------
export_df = pd.DataFrame({
    "Ticker": tickers,
    "GMV": np.array([w_gmv_dict.get(t, 0.0) for t in tickers]),
    "Tangency": np.array([w_tan_dict.get(t, 0.0) for t in tickers]),
    f"Gamma_{gamma:.2f}": w_gamma
})
st.download_button(
    "Download Weights CSV",
    data=export_df.to_csv(index=False),
    file_name="weights_true_constrained.csv",
    mime="text/csv"
)

st.caption(
    "Frontier curve drawn with target-return sweep (preference-free) for a smooth, faithful shape. "
    "The γ-portfolio overlays your investor risk tolerance. "
    "Log-return μ and sample covariance Σ are annualized; bounds match the sidebar settings."
)
