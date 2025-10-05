import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pypfopt import risk_models, EfficientFrontier
import cvxpy as cp

st.set_page_config(page_title="Efficient Frontier & CML", layout="wide")
st.title("Efficient Frontier & CML (Long/Short optional)")

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("Inputs")
    rf = st.number_input("Risk-free rate (annual, e.g. 0.02 = 2%)",
                         value=0.02, step=0.001, format="%.4f")
    gamma = st.number_input("Risk aversion γ (≥ 0)",
                            value=3.0, step=0.5, min_value=0.0, format="%.3f")
    log_returns = st.checkbox("Use log returns", value=True)
    freq = st.selectbox("Periods per year",
                        [252, 365, 12], index=0,
                        help="252=trading days, 365=daily calendar, 12=monthly")
    allow_short = st.checkbox("Allow shorting (long/short)", value=False)
    lower_bound = -1.0 if allow_short else 0.0
    upper_bound = 1.0

    uploaded = st.file_uploader(
        "Upload CSV of **daily prices** (first col: Date; other cols: tickers).",
        type=["csv"],
        help="Date must parse as YYYY-MM-DD (or similar)."
    )

# -------------------- Data load --------------------
@st.cache_data(show_spinner=False)
def load_prices(file) -> pd.DataFrame:
    df = pd.read_csv(file, parse_dates=["Date"]).set_index("Date").sort_index()
    df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)
    df = df.ffill().bfill()
    return df

if uploaded is None:
    st.info("Upload a CSV to continue. You can test with `data/example_prices.csv` in the repo.")
    st.stop()

prices = load_prices(uploaded)
if prices.shape[1] < 1:
    st.error("No ticker columns found after loading the CSV.")
    st.stop()

# -------------------- Returns (single source of truth) --------------------
if log_returns:
    rets = np.log(prices).diff().dropna()
else:
    rets = prices.pct_change().dropna()

# Annualized mean returns (μ) and covariance (Σ) from the SAME returns
mu = rets.mean() * freq                        # (vector)
Sigma = risk_models.sample_cov(rets, frequency=freq, returns_data=True)  # (matrix)
tickers = list(mu.index)

# -------------------- γ-portfolio solver (excess return + ½γ) --------------------
def solve_gamma(mu_vec, Sigma_mat, gamma, lb, ub, rf):
    """max (μ - rf)^T w - 0.5*γ * w^T Σ w  s.t. 1^T w=1, lb ≤ w ≤ ub"""
    n = len(mu_vec)
    w = cp.Variable(n)
    mu_excess = mu_vec - rf
    objective = cp.Maximize(mu_excess @ w - 0.5 * gamma * cp.quad_form(w, Sigma_mat))
    constraints = [cp.sum(w) == 1, w >= lb, w <= ub]
    prob = cp.Problem(objective, constraints)
    for solver in [cp.OSQP, cp.SCS, cp.ECOS]:
        try:
            prob.solve(solver=solver, verbose=False)
            if w.value is not None:
                return np.array(w.value).reshape(-1)
        except Exception:
            pass
    raise RuntimeError("CVXPY failed to solve the gamma-portfolio with available solvers.")

# -------------------- Frontier sampling --------------------
def compute_frontier(mu, Sigma, lb, ub, rf):
    # Sweep target returns over a modest, feasible band
    ret_min = float(mu.min())
    ret_max = float(mu.max() * 1.25)
    ret_grid = np.linspace(ret_min, ret_max, 60)

    vols, rets_out = [], []
    for r in ret_grid:
        ef = EfficientFrontier(mu, Sigma, weight_bounds=(lb, ub))
        try:
            ef.efficient_return(target_return=r)
            vol, ret, _ = ef.portfolio_performance(risk_free_rate=rf)
            vols.append(vol); rets_out.append(ret)
        except Exception:
            continue
    return np.array(vols), np.array(rets_out)

# -------------------- Key portfolios --------------------
# GMV
ef_gmv = EfficientFrontier(mu, Sigma, weight_bounds=(lower_bound, upper_bound))
w_gmv = ef_gmv.min_volatility()
gmv_vol, gmv_ret, _ = ef_gmv.portfolio_performance(risk_free_rate=rf)

# MSR (Tangency)
ef_msr = EfficientFrontier(mu, Sigma, weight_bounds=(lower_bound, upper_bound))
w_msr = ef_msr.max_sharpe(risk_free_rate=rf)
msr_vol, msr_ret, _ = ef_msr.portfolio_performance(risk_free_rate=rf)

# γ-portfolio
w_gamma = solve_gamma(mu.values, Sigma.values, gamma=gamma,
                      lb=lower_bound, ub=upper_bound, rf=rf)
gamma_ret = float(mu.values @ w_gamma)
gamma_vol = float(np.sqrt(w_gamma @ Sigma.values @ w_gamma))

# Frontier curve
ef_vols, ef_rets = compute_frontier(mu, Sigma, lower_bound, upper_bound, rf)

# -------------------- Plot --------------------
fig = plt.figure(figsize=(6.5, 4.5))
if ef_vols.size:
    plt.scatter(ef_vols, ef_rets, s=12, label="Efficient Frontier")
plt.scatter([gmv_vol], [gmv_ret], marker="D", label="GMV")
plt.scatter([msr_vol], [msr_ret], marker="*", s=140, label="MSR (Tangency)")
plt.scatter([gamma_vol], [gamma_ret], marker="o", label=f"Gamma (γ={gamma:.2f})")

# Capital Market Line (through RF and tangency)
xmax = max([msr_vol, ef_vols.max() if ef_vols.size else msr_vol]) * 1.2
x = np.linspace(0, xmax, 60)
cml = rf + (msr_ret - rf) * (x / msr_vol) if msr_vol > 1e-10 else np.full_like(x, np.nan)
plt.plot(x, cml, label="CML")

plt.xlabel("Volatility (σ)")
plt.ylabel("Expected Return (μ)")
plt.title("Efficient Frontier & Capital Market Line")
plt.legend(loc="best")
st.pyplot(fig, clear_figure=True)

# -------------------- Weights tables --------------------
def weights_to_df(weights_dict_or_vec, labels):
    if isinstance(weights_dict_or_vec, dict):
        wv = np.array([weights_dict_or_vec[t] for t in labels])
    else:
        wv = np.asarray(weights_dict_or_vec)
    df = pd.DataFrame({"Ticker": labels, "Weight": wv})
    df["Weight"] = df["Weight"].round(6)
    df = df[df["Weight"].abs() > 1e-8].sort_values("Weight", ascending=False).reset_index(drop=True)
    return df

st.markdown("### Portfolios & Weights")
c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("GMV")
    st.dataframe(weights_to_df(w_gmv, tickers), use_container_width=True)
with c2:
    st.subheader("MSR (Tangency)")
    st.dataframe(weights_to_df(w_msr, tickers), use_container_width=True)
with c3:
    st.subheader(f"Gamma (γ={gamma:.2f})")
    st.dataframe(weights_to_df(w_gamma, tickers), use_container_width=True)

# -------------------- Export combined weights --------------------
export_df = pd.DataFrame({
    "Ticker": tickers,
    "GMV": np.array([w_gmv[t] for t in tickers]),
    "MSR": np.array([w_msr[t] for t in tickers]),
    f"Gamma_{gamma:.2f}": w_gamma
})
st.download_button(
    "Download weights CSV",
    data=export_df.to_csv(index=False),
    file_name="weights.csv",
    mime="text/csv"
)

st.caption(
    "Notes: Upload daily prices (Date + tickers). μ and Σ are computed from the same returns "
    f"({'log' if log_returns else 'simple'}) and annualized with frequency={freq}. "
    "MSR = max Sharpe (tangency). γ-portfolio solves max (μ−rf)^T w − ½γ w^TΣw with ∑w=1 and bounds."
)
