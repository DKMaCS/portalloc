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
    n_frontier = st.slider("Frontier points", 30, 300, 160)
    include_cash = st.toggle("Include risk-free asset in γ optimization (γ on CML)", value=True)

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
mu = rets.mean() * freq                          # Series (annualized)
Sigma = risk_models.sample_cov(rets, frequency=freq, returns_data=True)  # DataFrame
mu_vec, Sigma_mat = mu.values, Sigma.values      # numpy arrays

# ---------------- Helpers ----------------
def make_ef(mu, Sigma):
    """Return an EfficientFrontier without box bounds (only ∑w=1).
    Falls back to very wide numeric bounds if needed."""
    try:
        return EfficientFrontier(mu, Sigma, weight_bounds=(None, None))
    except Exception:
        return EfficientFrontier(mu, Sigma, weight_bounds=(-1e9, 1e9))

def clean_weights_array_from_dict(dct, tickers, cutoff=1e-8):
    w = np.array([dct.get(t, 0.0) for t in tickers], dtype=float)
    w[np.abs(w) < cutoff] = 0.0
    s = w.sum()
    return w if s == 0 else w / s

def clean_weights_array(w, cutoff=1e-8):
    w = np.array(w, float).ravel()
    w[np.abs(w) < cutoff] = 0.0
    s = w.sum()
    return w if s == 0 else w / s

def portfolio_stats_risky_only(w_risky, mu_vec, Sigma_mat, rf):
    """Return,vol,Sharpe for a risky-only portfolio (∑w=1)."""
    ret = float(w_risky @ mu_vec)
    var = float(w_risky @ Sigma_mat @ w_risky)
    vol = float(np.sqrt(max(var, 0.0)))
    sharpe = (ret - rf) / max(vol, 1e-12)
    return ret, vol, sharpe

def portfolio_stats_with_cash(w0, w_risky, mu_vec, Sigma_mat, rf):
    """Return,vol,Sharpe when a cash weight w0 is allowed (w0 + 1' w_risky = 1)."""
    ret = float(rf * w0 + w_risky @ mu_vec)
    var = float(w_risky @ Sigma_mat @ w_risky)    # cash is riskless
    vol = float(np.sqrt(max(var, 0.0)))
    sharpe = (ret - rf) / max(vol, 1e-12) if vol > 0 else 0.0
    return ret, vol, sharpe

def solve_gamma_risky_only(mu_vec, Sigma_mat, rf, gamma):
    """max (μ−rf)^T w − ½γ w^T Σ w  s.t. ∑w=1 (no box bounds; risky-only)."""
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
                return np.asarray(w.value, dtype=float).reshape(-1), 0.0
        except Exception:
            continue
    raise RuntimeError("CVXPY failed to solve γ-portfolio (risky-only).")

def solve_gamma_with_cash(mu_vec, Sigma_mat, rf, gamma):
    """max rf*w0 + μ^T w − ½γ w^T Σ w s.t. w0 + 1'w = 1 (w0 is cash)."""
    n = len(mu_vec)
    w0 = cp.Variable()
    w = cp.Variable(n)
    objective = cp.Maximize(rf * w0 + mu_vec @ w - 0.5 * gamma * cp.quad_form(w, Sigma_mat))
    constraints = [w0 + cp.sum(w) == 1]
    prob = cp.Problem(objective, constraints)
    for solver in [cp.OSQP, cp.SCS, cp.ECOS]:
        try:
            prob.solve(solver=solver, verbose=False)
            if w.value is not None and w0.value is not None:
                return np.asarray(w.value, float).reshape(-1), float(w0.value)
        except Exception:
            continue
    raise RuntimeError("CVXPY failed to solve γ-portfolio (with cash).")

def compute_frontier_target_vol(mu, Sigma, rf, n_points, tan_vol_hint=None):
    """Sweep target volatility (efficient_risk) over a wide, robust range."""
    ef_gmv = make_ef(mu, Sigma)
    ef_gmv.min_volatility()
    gmv_ret, gmv_vol, _ = ef_gmv.portfolio_performance(risk_free_rate=rf)

    # Wide high-σ anchor so we cover well past tangency and most γ points
    single_asset_vol_max = float(np.sqrt(np.max(np.diag(Sigma.values))))
    hi_candidates = [
        1.50 * single_asset_vol_max,  # beyond any individual asset
        4.00 * gmv_vol,               # well past GMV
    ]
    if tan_vol_hint is not None and np.isfinite(tan_vol_hint):
        hi_candidates += [1.50 * tan_vol_hint]
    hi_vol = max(hi_candidates)

    sig_targets = np.linspace(gmv_vol, hi_vol, n_points)
    vols, rets = [], []
    last_w = None
    for s in sig_targets:
        ef = make_ef(mu, Sigma)
        try:
            ef.efficient_risk(target_volatility=s)
            w_dict = ef.clean_weights(cutoff=1e-10)
            w_vec = clean_weights_array_from_dict(w_dict, mu.index, cutoff=0.0)
            if last_w is not None and np.linalg.norm(w_vec - last_w, 1) <= 1e-10:
                continue
            ret, vol, _ = ef.portfolio_performance(risk_free_rate=rf)
            vols.append(vol); rets.append(ret); last_w = w_vec
        except Exception:
            continue
    return np.array(vols), np.array(rets), gmv_vol, gmv_ret

# ---------------- Key portfolios ----------------
# Tangency (max Sharpe) — risky frontier tangency
ef_tan = make_ef(mu, Sigma)
ef_tan.max_sharpe(risk_free_rate=rf)
w_tan_dict = ef_tan.clean_weights(cutoff=1e-10)
w_tan = clean_weights_array_from_dict(w_tan_dict, tickers, cutoff=1e-10)
tan_ret, tan_vol, tan_sharpe = ef_tan.portfolio_performance(risk_free_rate=rf)

# GMV (min variance) — risky frontier GMV
ef_gmv = make_ef(mu, Sigma)
ef_gmv.min_volatility()
w_gmv_dict = ef_gmv.clean_weights(cutoff=1e-10)
w_gmv = clean_weights_array_from_dict(w_gmv_dict, tickers, cutoff=1e-10)
gmv_ret, gmv_vol, _ = ef_gmv.portfolio_performance(risk_free_rate=rf)

# Frontier curve (preference-free) — σ-sweep, extended well past tangency
ef_vols, ef_rets, gmv_vol, gmv_ret = compute_frontier_target_vol(
    mu, Sigma, rf, n_frontier, tan_vol_hint=tan_vol
)

# γ portfolio
if include_cash:
    w_gamma_risky, w0_gamma = solve_gamma_with_cash(mu_vec, Sigma_mat, rf, gamma)
    w_gamma_risky = clean_weights_array(w_gamma_risky)         # risky weights (sum to 1 - w0)
    gamma_ret, gamma_vol, gamma_sharpe = portfolio_stats_with_cash(w0_gamma, w_gamma_risky, mu_vec, Sigma_mat, rf)
    gamma_label = f"Γ-Portfolio (γ={gamma:.2f}, cash allowed)"
else:
    w_gamma_risky, _ = solve_gamma_risky_only(mu_vec, Sigma_mat, rf, gamma)
    w_gamma_risky = clean_weights_array(w_gamma_risky)         # risky weights (sum to 1)
    gamma_ret, gamma_vol, gamma_sharpe = portfolio_stats_risky_only(w_gamma_risky, mu_vec, Sigma_mat, rf)
    w0_gamma = 0.0
    gamma_label = f"Γ-Portfolio (γ={gamma:.2f}, risky-only)"

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(5.5, 3.8))

if ef_vols.size:
    order = np.argsort(ef_vols)
    ax.plot(ef_vols[order], ef_rets[order], lw=1.6, alpha=0.95, label="Efficient Frontier (risky)")

# key points
ax.scatter([gmv_vol], [gmv_ret], marker="D", label="GMV")
ax.scatter([tan_vol], [tan_ret], marker="*", s=140, label="Tangency")
ax.scatter([gamma_vol], [gamma_ret], marker="o", label=gamma_label)

# CML through tangency
xmax = 1.05 * max(tan_vol, gamma_vol, (ef_vols.max() if ef_vols.size else 0.0))
x = np.linspace(0, xmax, 200)
cml = rf + (tan_ret - rf) * (x / max(tan_vol, 1e-12))
ax.plot(x, cml, linestyle="--", label="CML")

ax.set_xlim(0, xmax)
ax.margins(y=0.05)
ax.set_xlabel("Volatility (σ, annualized)")
ax.set_ylabel("Expected Return (μ, annualized)")
ax.set_title("Efficient Frontier (risky), CML, and γ-Portfolio")
ax.legend(loc="best")
st.pyplot(fig, clear_figure=True)

# ---------------- Tables ----------------
st.markdown("### Portfolio Performance Summary")
summary_rows = [
    ["GMV", gmv_ret, gmv_vol, (gmv_ret - rf) / max(gmv_vol, 1e-12)],
    ["Tangency", tan_ret, tan_vol, (tan_ret - rf) / max(tan_vol, 1e-12)],
    [gamma_label, gamma_ret, gamma_vol, gamma_sharpe],
]
summary_df = pd.DataFrame(summary_rows, columns=["Portfolio", "Expected Return", "Volatility", "Sharpe Ratio"])
st.dataframe(summary_df.round(6), use_container_width=True)

def weights_df_from_array(weights, tickers, title):
    df = pd.DataFrame({"Ticker": tickers, "Weight": np.array(weights, float)})
    df["Weight"] = df["Weight"].round(6)
    df = df[df["Weight"].abs() > 1e-8].sort_values("Weight", ascending=False).reset_index(drop=True)
    st.subheader(title)
    st.dataframe(df, use_container_width=True)

st.markdown("### Portfolio Weights")
c1, c2, c3 = st.columns(3)
with c1:
    weights_df_from_array([w_gmv_dict.get(t, 0.0) for t in tickers], tickers, "GMV (risky)")
with c2:
    weights_df_from_array([w_tan_dict.get(t, 0.0) for t in tickers], tickers, "Tangency (risky)")
with c3:
    if include_cash:
        st.subheader(f"Γ (with cash, γ={gamma:.2f})")
        st.write(f"Risk-free weight: **{w0_gamma:.6f}**")
        weights_df_from_array(w_gamma_risky, tickers, "Γ risky weights")
    else:
        weights_df_from_array(w_gamma_risky, tickers, f"Γ (risky-only, γ={gamma:.2f})")

# ---------------- Export ----------------
export_df = pd.DataFrame({
    "Ticker": tickers,
    "GMV": np.array([w_gmv_dict.get(t, 0.0) for t in tickers]),
    "Tangency": np.array([w_tan_dict.get(t, 0.0) for t in tickers]),
    f"Gamma_risky_{gamma:.2f}": w_gamma_risky,
})
if include_cash:
    export_df.insert(1, "Gamma_cash_weight", w0_gamma)  # single value will be broadcast if displayed; CSV keeps column
st.download_button(
    "Download Weights CSV",
    data=export_df.to_csv(index=False),
    file_name="weights_unconstrained.csv",
    mime="text/csv"
)

st.caption(
    "Frontier is for risky assets only (no box bounds), budget ∑w_risky=1 for EF; "
    "CML is the line from rf through the tangency portfolio. "
    "Toggle adds a cash weight to the γ optimization so γ sits on the CML. "
    "With risky-only γ, the point lies on the risky frontier and below the CML past tangency."
)
