import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# Page setup and visual theme
# ------------------------------------------------------------
st.set_page_config(
    page_title="StockPulse | Market Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --bg: #080d18;
        --panel: #101a2b;
        --panel2: #121f33;
        --line: rgba(148, 163, 184, .16);
        --muted: #91a1b8;
        --text: #edf4ff;
        --green: #25d695;
        --red: #ff647c;
        --blue: #69a7ff;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(37, 214, 149, .10), transparent 32%),
            radial-gradient(ellipse at 95% 8%, rgba(105, 167, 255, .10), transparent 28%),
            var(--bg);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: rgba(8,13,24,.75); }
    [data-testid="stSidebar"] {
        background: #0c1423;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] * { color: var(--text); }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: var(--text) !important; }
    p, label, .stMarkdown { color: #c7d3e5; }
    .hero {
        padding: 26px 30px;
        border: 1px solid rgba(105,167,255,.22);
        border-radius: 22px;
        background: linear-gradient(115deg, rgba(19,39,59,.98), rgba(16,26,43,.92) 58%, rgba(17,48,49,.88));
        margin-bottom: 22px;
        box-shadow: 0 16px 48px rgba(0,0,0,.20);
    }
    .eyebrow {
        color: var(--green);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .hero-title { font-family: 'Space Grotesk', sans-serif; font-size: 32px; font-weight: 700; color: #f4f8ff; line-height: 1.15; }
    .hero-sub { color: #a8bad0; font-size: 14px; margin-top: 8px; }
    .metric-card {
        background: linear-gradient(145deg, rgba(20,33,53,.98), rgba(14,24,40,.98));
        border: 1px solid var(--line);
        border-radius: 17px;
        padding: 18px 19px;
        min-height: 112px;
        box-shadow: 0 8px 24px rgba(0,0,0,.13);
    }
    .metric-label { color: #8fa2bc; font-size: 11px; font-weight: 700; letter-spacing: 1.1px; text-transform: uppercase; }
    .metric-value { color: #f2f7ff; font-family: 'Space Grotesk', sans-serif; font-size: 25px; font-weight: 700; margin-top: 9px; }
    .metric-note { color: #8fa2bc; font-size: 11px; margin-top: 3px; }
    .section-head { font-family: 'Space Grotesk', sans-serif; font-size: 19px; font-weight: 700; color: #eef5ff; margin: 22px 0 10px; }
    .subtle { color: #91a1b8; font-size: 12px; }
    .pill {
        display: inline-block; padding: 6px 11px; border-radius: 99px;
        font-size: 11px; font-weight: 700; letter-spacing: .7px;
        background: rgba(37,214,149,.12); color: #54e5ad; border: 1px solid rgba(37,214,149,.25);
    }
    .warning {
        background: rgba(255,190,92,.08); border: 1px solid rgba(255,190,92,.22);
        color: #e8c99b; padding: 12px 15px; border-radius: 12px; font-size: 12px;
    }
    div.stButton > button {
        width: 100%; border: 0; border-radius: 12px; padding: 11px 16px;
        background: linear-gradient(90deg, #19bd85, #24d69a); color: #061a17;
        font-weight: 800; transition: .2s ease;
    }
    div.stButton > button:hover { filter: brightness(1.08); color: #061a17; border: 0; }
    div[data-baseweb="input"] input, div[data-baseweb="select"] > div {
        background: #111e31 !important; color: #edf4ff !important; border-color: #293a52 !important;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid var(--line); }
    .stTabs [data-baseweb="tab"] { color: #91a1b8; background: transparent; }
    .stTabs [aria-selected="true"] { color: #4be0a9 !important; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📈 STOCKPULSE")
    st.caption("MARKET INTELLIGENCE")
    st.markdown("---")
    st.markdown("### Analysis settings")
    ticker = st.text_input("Stock ticker", value="AAPL").strip().upper()
    period = st.selectbox("Historical period", ["1y", "2y", "5y"], index=1)
    st.caption("Examples: AAPL · MSFT · TSLA · RELIANCE.NS")
    analyze_button = st.button("Analyze stock  →", use_container_width=True)
    st.markdown("---")
    st.markdown("**Model toolkit**")
    st.markdown("`PCA`  ·  `Gradient Descent`")
    st.markdown("`Eigenvectors`  ·  `Differential Equation`")
    st.markdown("---")
    st.caption("Educational project • Not investment advice")

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">● Market analytics workspace</div>
      <div class="hero-title">Stock Market<br>Trend Prediction</div>
      <div class="hero-sub">Explore historical market behaviour, dimensionality reduction and a simple machine-learning trend estimate.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

# ------------------------------------------------------------
# Data and model functions
# ------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def get_stock_data(symbol, selected_period):
    data = yf.download(symbol, period=selected_period, interval="1d", auto_adjust=True, progress=False)
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in data.columns for col in needed):
        return pd.DataFrame()
    return data[needed].dropna()

def build_features(data):
    frame = data.copy()
    frame["daily_return"] = frame["Close"].pct_change()
    frame["momentum_5d"] = frame["Close"].pct_change(5)
    frame["volatility_5d"] = frame["daily_return"].rolling(5).std()
    frame["price_range"] = (frame["High"] - frame["Low"]) / frame["Close"]
    frame["volume_change"] = frame["Volume"].pct_change()
    frame["target"] = frame["daily_return"].shift(-1)
    return frame.replace([np.inf, -np.inf], np.nan).dropna()

def gradient_descent_fit(X, y, learning_rate=0.03, epochs=2500):
    Xb = np.c_[np.ones(len(X)), X]
    weights = np.zeros(Xb.shape[1])
    n = len(y)
    for _ in range(epochs):
        error = Xb @ weights - y
        gradient = (2 / n) * (Xb.T @ error)
        weights -= learning_rate * gradient
    return weights

def predict_with_weights(X, weights):
    Xb = np.c_[np.ones(len(X)), X]
    return Xb @ weights

def run_analysis(symbol, selected_period):
    raw = get_stock_data(symbol, selected_period)
    if raw.empty or len(raw) < 100:
        raise ValueError("Not enough stock data was returned. Check the ticker and try another period.")

    frame = build_features(raw)
    feature_cols = ["daily_return", "momentum_5d", "volatility_5d", "price_range", "volume_change"]
    X = frame[feature_cols].to_numpy(dtype=float)
    y = frame["target"].to_numpy(dtype=float)

    split = int(len(frame) * 0.8)
    if split < 30 or len(frame) - split < 10:
        raise ValueError("Not enough observations for a train/test split.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[:split])
    X_test = scaler.transform(X[split:])
    y_train, y_test = y[:split], y[split:]

    pca = PCA(n_components=2)
    train_pca = pca.fit_transform(X_train)
    all_scaled = scaler.transform(X)
    all_pca = pca.transform(all_scaled)

    weights = gradient_descent_fit(X_train, y_train)
    test_predictions = predict_with_weights(X_test, weights)

    mae = float(np.mean(np.abs(test_predictions - y_test)))
    mse = float(np.mean((test_predictions - y_test) ** 2))
    rmse = float(np.sqrt(mse))
    directional_accuracy = float(np.mean(np.sign(test_predictions) == np.sign(y_test)) * 100)

    # Fit a final model on all available observations for the next-session estimate.
    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X)
    final_weights = gradient_descent_fit(X_all_scaled, y)
    latest_features = frame[feature_cols].to_numpy(dtype=float)[-1:].reshape(1, -1)
    latest_scaled = final_scaler.transform(latest_features)
    next_return = float(predict_with_weights(latest_scaled, final_weights)[0])
    latest_price = float(raw["Close"].iloc[-1])
    estimated_price = latest_price * (1 + next_return)

    if next_return > 0.001:
        trend = "UPWARD"
    elif next_return < -0.001:
        trend = "DOWNWARD"
    else:
        trend = "SIDEWAYS"

    # Illustrative 30-day exponential projection based on the mean daily return.
    k = float(np.clip(np.mean(y_train), -0.02, 0.02))
    days = np.arange(0, 31)
    projection = latest_price * np.exp(k * days)

    return {
        "raw": raw, "frame": frame, "feature_cols": feature_cols,
        "all_pca": all_pca, "split": split, "test_predictions": test_predictions,
        "y_test": y_test, "mae": mae, "mse": mse, "rmse": rmse,
        "accuracy": directional_accuracy, "variance": float(pca.explained_variance_ratio_.sum() * 100),
        "latest_price": latest_price, "estimated_price": estimated_price,
        "next_return": next_return, "trend": trend,
        "days": days, "projection": projection,
    }

if analyze_button:
    if not ticker:
        st.error("Enter a stock ticker first.")
    else:
        with st.spinner(f"Analysing {ticker} market data..."):
            try:
                st.session_state.analysis_result = run_analysis(ticker, period)
                st.session_state.last_ticker = ticker
            except Exception as exc:
                st.session_state.analysis_result = None
                st.error(f"Could not analyse {ticker}. {exc}")

result = st.session_state.analysis_result

if result is None:
    st.markdown('<div class="section-head">Welcome to your market workspace</div>', unsafe_allow_html=True)
    st.markdown(
        """
        Enter a ticker in the sidebar and select a historical period, then click **Analyze stock**.
        The dashboard will calculate model metrics and display the visual analysis.
        """)
    st.info("Try AAPL (Apple), MSFT (Microsoft), TSLA (Tesla) or RELIANCE.NS (Reliance Industries).")
    st.markdown(
        '<div class="warning">This is an educational machine-learning demonstration. Short-term stock movements are difficult to predict; results are not financial advice.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ------------------------------------------------------------
# Dashboard results
# ------------------------------------------------------------
raw = result["raw"]
last_ticker = st.session_state.get("last_ticker", ticker)
trend = result["trend"]
trend_color = "#25d695" if trend == "UPWARD" else ("#ff647c" if trend == "DOWNWARD" else "#69a7ff")
currency_symbol = "₹" if last_ticker.endswith(".NS") or last_ticker.endswith(".BO") else "$"

st.markdown(
    f'<div class="section-head">Market overview <span class="pill">{last_ticker}</span></div>',
    unsafe_allow_html=True,
)
st.caption(f"Latest available close: {raw.index[-1].strftime('%d %b %Y')}  ·  Historical window: {period}")

c1, c2, c3, c4 = st.columns(4)
cards = [
    (c1, "LATEST CLOSE", f"{currency_symbol}{result['latest_price']:,.2f}", "Most recent adjusted close"),
    (c2, "EST. NEXT PRICE", f"{currency_symbol}{result['estimated_price']:,.2f}", f"{result['next_return'] * 100:+.3f}% estimated return"),
    (c3, "PREDICTED TREND", trend, "Model direction estimate"),
    (c4, "DIRECTIONAL ACCURACY", f"{result['accuracy']:.2f}%", "Held-out test period"),
]
for col, label, value, note in cards:
    with col:
        value_color = trend_color if label == "PREDICTED TREND" else "#f2f7ff"
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value" style="color:{value_color}">{value}</div>'
            f'<div class="metric-note">{note}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown('<div class="section-head">Model performance</div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
for col, label, value in [
    (m1, "MAE", f"{result['mae']:.6f}"),
    (m2, "RMSE", f"{result['rmse']:.6f}"),
    (m3, "MSE", f"{result['mse']:.8f}"),
    (m4, "PCA VARIANCE RETAINED", f"{result['variance']:.2f}%"),
]:
    with col:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

# Chart theme helper
def style_axis(ax, title):
    ax.set_title(title, color="#edf4ff", fontsize=13, fontweight="bold", pad=14)
    ax.set_facecolor("#101a2b")
    ax.tick_params(colors="#91a1b8", labelsize=9)
    ax.grid(True, color="#26364b", alpha=0.55, linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_color("#26364b")
    ax.xaxis.label.set_color("#91a1b8")
    ax.yaxis.label.set_color("#91a1b8")

st.markdown('<div class="section-head">Market intelligence</div>', unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["Price history", "PCA map", "Model accuracy", "30-day projection"])

with tab1:
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor("#101a2b")
    ax.plot(raw.index, raw["Close"], color="#25d695", linewidth=2)
    ax.fill_between(raw.index, raw["Close"].to_numpy(), raw["Close"].min(), color="#25d695", alpha=0.08)
    style_axis(ax, f"{last_ticker} · Historical closing price")
    ax.set_ylabel("Adjusted price")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor("#101a2b")
    scatter = ax.scatter(result["all_pca"][:, 0], result["all_pca"][:, 1],
                          c=np.arange(len(result["all_pca"])), cmap="winter", s=16, alpha=0.75)
    style_axis(ax, "PCA · Five features reduced to two components")
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    fig.colorbar(scatter, ax=ax, label="Observation order")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(f"The first two principal components retain {result['variance']:.2f}% of the feature variance.")

with tab3:
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor("#101a2b")
    x_axis = np.arange(len(result["y_test"]))
    ax.plot(x_axis, result["y_test"] * 100, color="#69a7ff", linewidth=1.7, label="Actual")
    ax.plot(x_axis, result["test_predictions"] * 100, color="#25d695", linewidth=1.5, alpha=0.9, label="Predicted")
    style_axis(ax, "Actual vs predicted next-day returns")
    ax.set_xlabel("Test observation")
    ax.set_ylabel("Daily return (%)")
    ax.legend(facecolor="#142238", edgecolor="#26364b", labelcolor="#edf4ff")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tab4:
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor("#101a2b")
    ax.plot(result["days"], result["projection"], color="#25d695", linewidth=2.2)
    ax.fill_between(result["days"], result["projection"], result["projection"].min(), color="#25d695", alpha=0.08)
    style_axis(ax, "Illustrative exponential price curve")
    ax.set_xlabel("Days from latest close")
    ax.set_ylabel("Illustrative price")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("This curve uses a simplified constant-growth differential equation. It is an illustration, not a dependable price forecast.")

with st.expander("Mathematical methods used"):
    st.markdown(r"""
    **1. Daily return**

    \[
    r_t = \frac{P_t-P_{t-1}}{P_{t-1}}
    \]

    **2. PCA and eigenvectors**

    \[
    Cv = \lambda v
    \]

    PCA reduces five engineered market features to two principal components.

    **3. Gradient descent**

    \[
    \theta_{\mathrm{new}} = \theta_{\mathrm{old}}-\alpha\nabla J(\theta)
    \]

    The model updates its weights to reduce mean squared error.

    **4. Differential equation**

    \[
    \frac{dP}{dt}=kP,\qquad P(t)=P_0e^{kt}
    \]
    """)

st.markdown(
    '<div class="warning">Important: This is a student educational prototype. A directional accuracy near 50% indicates limited predictive power. Stock prices are uncertain; do not use these estimates to make investment decisions.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtle" style="text-align:center; margin-top:24px;">STOCKPULSE · Built with Python & Streamlit · Academic project</p>',
    unsafe_allow_html=True,
)
