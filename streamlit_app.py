# streamlit_app.py
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import cvxpy as cp
from pypfopt import risk_models, EfficientFrontier

st.set_page_config(page_title="Efficient Frontier & CML — Unconstrained Weights", layout="wide")
st.title("Efficient Frontier & CML (Unconstrained Weights; ∑w = 1)")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Inputs")
    rf = st.number_input("Risk-free rate (annual)", value=0.02, step=0.001, format="%.4f")
    gamma = st.number_input("Risk aversion γ (≥ 0)", value=300.0, step=10.0, min_value=0.0, format="%.2f")
    freq = st.selectbox("Periods per year", [252, 52, 12], index=0)
    n_frontier = st.slider("Frontier points (γ sweep)", 30, 300, 140)
    g_min = st.number_input("γ min (log sweep)", value=1e-2, format="%.4g")
    g_max = st.number_input("γ max (log sweep)", value=1e4, format="%.4g")

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

# ---------------- Helpers ----------------
def make_ef(mu, Sigma):
    try:
        return EfficientFrontier(mu, Sigma, weight_bounds=(None, None))
    except Exception:
        return EfficientFrontier(mu, Sigma, weight_bounds=(-1e9, 1e9))

def clean_weights_array_from_dict(dct, tickers, cutoff=1e-6):
    w = np.array([dct.get(t, 0.0) for t in tickers], dtype=float)
    w[np.abs(w) < cutoff] = 0.0
    s = w.sum()
    return w if s == 0 else w / s

def clean_weights_array(w, cutoff=1e-6):
    w = np.array(w, float).ravel()
    w[np.abs(w) < cutoff] = 0.0
    s = w.sum()
    return w if s == 0 else w / s

def portfolio_stats(w, mu_vec, Sigma_mat, rf):
    ret = float(w @ mu_vec)
    vol = float(np.sqrt(max(w @ Sigma_mat @ w, 0.0)))
    sharpe = (ret - rf) / max(vol, 1e-12)
    return ret, vol, sharpe

# ---- Your γ-optimizer (keep exactly this objective) ----
def solve_gamma_portfolio(mu_vec, Sigma_mat, rf, gamma):
    """max (μ−rf)^T w − ½γ w^T Σ w  s.t. ∑w=1 (no box bounds)"""
    n = len(mu_vec)
    w = cp.Variable(n)
    mu_ex = mu_vec - rf
    objective = cp.Maximize(mu_ex @ w - 0.5 * gamma * cp.quad_form(w, Sigma_mat))
    constraints = [cp.sum(w) == 1]
    prob = cp.Problem(objective, constraints)
    for solver in [cp.OSQP, cp.SCS, cp.ECOS]:
        try:
            prob.solve(solver=solver, verbose=False)
            if w.value is not None:
                return np.asarray(w.value, dtype=float).reshape(-1)
        except Exception:
            pass
    raise RuntimeError("CVXPY failed to solve γ-portfolio (unconstrained).")

# ---- Build the EF by sweeping γ using the SAME solver/objective ----
def compute_frontier_by_gamma(mu_vec, Sigma_mat, rf, g_min, g_max, n_points):
    gammas = np.logspace(np.log10(g_max), np.log10(g_min), n_points)  # high→low risk aversion
    vols, rets = [], []
    last_w = None
    for g in gammas:
        w = solve_gamma_portfolio(mu_vec, Sigma_mat, rf, g)
        # skip near-duplicate corners
        if last_w is not None and np.linalg.norm(w - last_w, 1) <= 1e-8:
            continue
        r, s, _ = portfolio_stats(w, mu_vec, Sigma_mat, rf)
        rets.append(r); vols.append(s); last_w = w
    vols, rets = np.array(vols), np.array(rets)
    order = np.argsort(vols)
    return vols[order], rets[order]

# ---------------- Key portfolios ----------------
# γ portfolio (investor preference)
w_gamma = solve_gamma_portfolio(mu_vec, Sigma_mat, rf, gamma)
w_gamma = clean_weights_array(w_gamma, cutoff=1e-6)
gamma_ret, gamma_vol, gamma_sharpe = portfolio_stats(w_gamma, mu_vec, Sigma_mat, rf)

# Tangency (max Sharpe) — just as a reference marker
ef_tan = make_ef(mu, Sigma)
ef_tan.max_sharpe(risk_free_rate=rf)
w_tan_dict = ef_tan.clean_weights(cutoff=1e-8)
w_tan = clean_weights_array_from_dict(w_tan_dict, tickers, cutoff=1e-8)
tan_ret, tan_vol, tan_sharpe = ef_tan.portfolio_performance(risk_free_rate=rf)

# GMV (min variance) — reference marker
ef_gmv = make_ef(mu, Sigma)
ef_gmv.min_volatility()
w_gmv_dict = ef_gmv.clean_weights(cutoff=1e-8)
w_gmv = clean_weights_array_from_dict(w_gmv_dict, tickers, cutoff=1e-8)
gmv_ret, gmv_vol, _ = ef_gmv.portfolio_performance(risk_free_rate=rf)

# Frontier built by γ-sweep (so γ point lies on it)
ef_vols, ef_rets = compute_frontier_by_gamma(mu_vec, Sigma_mat, rf, g_min=g_min, g_max=g_max, n_points=n_frontier)

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(5.0, 3.6), dpi=120)

if ef_vols.size:
    ax.plot(ef_vols, ef_rets, lw=1.8, alpha=0.95, label="Efficient Frontier (γ sweep)")

# key points
ax.scatter([gmv_vol], [gmv_ret], marker="D", label="GMV")
ax.scatter([tan_vol], [tan_ret], marker="*", s=120, label="Tangency")
ax.scatter([gamma_vol], [gamma_ret], marker="o", label=f"Γ-Portfolio (γ={gamma:.2f})")

# CML (through tangency) for reference only
xmax = 1.05 * max(tan_vol, gamma_vol, (ef_vols.max() if ef_vols.size else 0.0))
x = np.linspace(0, xmax, 200)
cml = rf + (tan_ret - rf) * (x / max(tan_vol, 1e-12))
ax.plot(x, cml, linestyle="--", label="CML")

ax.set_xlim(0, xmax)
ax.margins(y=0.05)
ax.set_xlabel("Volatility (σ, annualized)")
ax.set_ylabel("Expected Return (μ, annualized)")
ax.set_title("Efficient Frontier — Unconstrained Weights (∑w=1) traced by γ")
ax.legend(loc="best", fontsize=8)
st.pyplot(fig, clear_figure=True, use_container_width=False)

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
    st.subheader(f"Γ-Portfolio (γ={gamma:.2f})")
    st.dataframe(weights_df(w_gamma, tickers), use_container_width=True)

# ---------------- Export ----------------
export_df = pd.DataFrame({
    "Ticker": tickers,
    "GMV": np.array([w_gmv_dict.get(t, 0.0) for t in tickers]),
    "Tangency": np.array([w_tan_dict.get(t, 0.0) for t in tickers]),
    f"Gamma_{gamma:.2f}": w_gamma,
})
st.download_button(
    "Download Weights CSV",
    data=export_df.to_csv(index=False),
    file_name="weights_unconstrained_gamma_frontier.csv",
    mime="text/csv"
)

st.caption(
    "The blue curve is produced by sweeping γ with the same objective used for the Γ-portfolio. "
    "Therefore the green point lies on the frontier (up to numerical tolerance). "
    "If the γ point still looks farther right, increase the γ sweep range or x-axis limit."
)
