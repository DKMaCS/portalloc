import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import cvxpy as cp
from pypfopt import risk_models, EfficientFrontier

st.set_page_config(page_title="Efficient Frontier & CML — True Constrained", layout="wide")
st.title("Efficient Frontier & CML (True Constrained γ-Formulation)")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Inputs")
    rf = st.number_input("Risk-free rate (annual, e.g. 0.02 = 2%)", value=0.02, step=0.001, format="%.4f")
    gamma = st.number_input("Risk aversion γ (≥ 0)", value=300.0, step=10.0, min_value=0.0, format="%.2f")
    freq = st.selectbox("Periods per year", [252, 365, 12], index=0)
    allow_short = st.checkbox("Allow shorting (long/short)", value=False)
    n_frontier = st.slider("Frontier points", 30, 200, 100)
    lower_bound = -1.0 if allow_short else 0.0
    upper_bound = 1.0

    uploaded = st.file_uploader(
        "Upload CSV of **daily prices** (first col: Date; other cols: tickers).",
        type=["csv"],
        help="Date must parse as YYYY-MM-DD (or similar)."
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

# ---------------- Returns & Covariance ----------------
rets = np.log(prices).diff().dropna()
mu = rets.mean() * freq
Sigma = risk_models.sample_cov(rets, frequency=freq, returns_data=True)

# ---------------- Solvers ----------------
def solve_gamma_portfolio(mu_vec, Sigma_mat, rf, gamma, lb, ub):
    """max (μ−rf)^T w − ½γ w^T Σ w  s.t. ∑w=1, bounds"""
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
                return np.array(w.value).reshape(-1)
        except Exception:
            pass
    raise RuntimeError("CVXPY failed to solve γ-portfolio.")

def portfolio_stats(w, mu_vec, Sigma_mat, rf):
    ret = float(w @ mu_vec)
    vol = float(np.sqrt(max(w @ Sigma_mat @ w, 0)))
    sharpe = (ret - rf) / max(vol, 1e-12)
    return ret, vol, sharpe

def compute_frontier_gamma(mu_vec, Sigma_mat, rf, lb, ub, n_points):
    gammas = np.logspace(-2, 3, n_points)
    rets, vols = [], []
    for g in gammas:
        try:
            w = solve_gamma_portfolio(mu_vec, Sigma_mat, rf, g, lb, ub)
            r, v, _ = portfolio_stats(w, mu_vec, Sigma_mat, rf)
            rets.append(r)
            vols.append(v)
        except Exception:
            continue
    return np.array(vols), np.array(rets)

# ---------------- Key Portfolios ----------------
mu_vec, Sigma_mat = mu.values, Sigma.values

# γ-portfolio (user-selected)
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

# ---------------- Frontier (γ sweep) ----------------
ef_vols, ef_rets = compute_frontier_gamma(mu_vec, Sigma_mat, rf, lower_bound, upper_bound, n_frontier)

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(ef_vols, ef_rets, s=12, label="Efficient Frontier (γ-sweep)")
ax.scatter([gmv_vol], [gmv_ret], marker="D", label="GMV")
ax.scatter([tan_vol], [tan_ret], marker="*", s=140, label="Tangency")
ax.scatter([gamma_vol], [gamma_ret], marker="o", label=f"Γ-Portfolio (γ={gamma:.1f})")

# Capital Market Line (CML)
xmax = max([tan_vol, ef_vols.max() if ef_vols.size else tan_vol]) * 1.1
x = np.linspace(0, xmax, 100)
cml = rf + (tan_ret - rf) * (x / tan_vol)
ax.plot(x, cml, linestyle="--", label="CML")

ax.set_xlabel("Volatility (σ, annualized)")
ax.set_ylabel("Expected Return (μ, annualized)")
ax.set_title("Efficient Frontier — True Constrained γ-Formulation")
ax.legend(loc="best")
st.pyplot(fig, clear_figure=True)

# ---------------- Performance Summary ----------------
st.markdown("### Portfolio Performance Summary")

summary_df = pd.DataFrame({
    "Portfolio": ["GMV", "Tangency", f"Gamma (γ={gamma:.1f})"],
    "Expected Return": [gmv_ret, tan_ret, gamma_ret],
    "Volatility": [gmv_vol, tan_vol, gamma_vol],
    "Sharpe Ratio": [
        (gmv_ret - rf) / gmv_vol,
        (tan_ret - rf) / tan_vol,
        (gamma_ret - rf) / gamma_vol,
    ]
})
st.dataframe(summary_df.round(4), use_container_width=True)

# ---------------- Weights Tables ----------------
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
    st.subheader(f"Γ-Portfolio (γ={gamma:.1f})")
    st.dataframe(weights_df(w_gamma, tickers), use_container_width=True)

# ---------------- Export ----------------
export_df = pd.DataFrame({
    "Ticker": tickers,
    "GMV": np.array([w_gmv_dict[t] for t in tickers]),
    "Tangency": np.array([w_tan_dict[t] for t in tickers]),
    f"Gamma_{gamma:.1f}": w_gamma
})
st.download_button(
    "Download Weights CSV",
    data=export_df.to_csv(index=False),
    file_name="weights_true_constrained.csv",
    mime="text/csv"
)

st.caption(
    "μ and Σ computed from log returns (annualized). Frontier and γ-portfolio solved via "
    "utility maximization max(μ−rf)^T w − ½γ w^TΣw with equality and bound constraints — "
    "identical to the Colab true-constrained formulation. Sharpe ratios show excess return per unit of risk."
)
