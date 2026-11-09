import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Stock Market Trend Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Market Trend Prediction System")
st.caption("Mathematical modelling using PCA, eigenvectors, gradient descent and differential equations")

st.info(
    "Educational demonstration only. Stock-market predictions are uncertain "
    "and this application is not financial advice."
)

with st.sidebar:
    st.header("Analysis settings")
    ticker_text = st.text_input("Stock ticker", "AAPL").strip().upper()
    period = st.selectbox("Historical period", ["1y", "2y", "5y"], index=1)
    run = st.button("Analyze stock", type="primary", use_container_width=True)

def get_stock_frame(ticker, period):
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError("No data found. Check the ticker symbol.")

    if isinstance(df.columns, pd.MultiIndex):
        found = False
        for level in range(df.columns.nlevels):
            if ticker in df.columns.get_level_values(level):
                df = df.xs(ticker, axis=1, level=level)
                found = True
                break
        if not found:
            df.columns = df.columns.get_level_values(0)

    required = ["Close", "High", "Low", "Volume"]
    if not all(c in df.columns for c in required):
        raise ValueError("Yahoo Finance did not return all required columns.")
    return df.dropna(subset=["Close"])

def analyze(df):
    close = df["Close"].dropna()
    features = pd.DataFrame(index=df.index)
    daily = close.pct_change()
    features["Daily_Return"] = daily
    features["Momentum"] = close.pct_change(5)
    features["Volatility"] = daily.rolling(5).std()
    features["Price_Range"] = (df["High"] - df["Low"]) / close
    features["Volume_Change"] = df["Volume"].pct_change()
    features = features.replace([np.inf, -np.inf], np.nan).dropna()

    target = daily.shift(-1)
    valid = target.reindex(features.index).dropna()
    X = features.loc[valid.index]
    y = valid

    if len(X) < 50:
        raise ValueError("Not enough historical data. Try a longer period.")

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split].values, y.iloc[split:].values

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train)
    test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=2)
    train_pca = pca.fit_transform(train_scaled)
    test_pca = pca.transform(test_scaled)

    # Gradient descent minimizes mean squared error.
    weights = np.zeros(2)
    bias = 0.0
    learning_rate = 0.01
    for _ in range(2000):
        pred = train_pca @ weights + bias
        error = pred - y_train
        dw = 2 * (train_pca.T @ error) / len(y_train)
        db = 2 * np.mean(error)
        weights -= learning_rate * dw
        bias -= learning_rate * db

    test_pred = test_pca @ weights + bias
    mse = float(np.mean((test_pred - y_test) ** 2))
    mae = float(np.mean(np.abs(test_pred - y_test)))
    rmse = float(np.sqrt(mse))
    directional = float(np.mean(np.sign(test_pred) == np.sign(y_test)) * 100)

    # Latest available feature row is used for the next-session estimate.
    latest_scaled = scaler.transform(features.iloc[[-1]])
    latest_pca = pca.transform(latest_scaled)
    next_return = float((latest_pca @ weights + bias).item())

    latest_price = float(close.iloc[-1])
    estimated_price = latest_price * (1 + next_return)

    if next_return > 0.001:
        trend = "UPWARD"
    elif next_return < -0.001:
        trend = "DOWNWARD"
    else:
        trend = "SIDEWAYS"

    # Illustrative constant-growth differential-equation projection.
    k = float(y_train.mean())
    days = np.arange(31)
    projection = latest_price * np.exp(k * days)

    return {
        "close": close, "latest_price": latest_price,
        "estimated_price": estimated_price, "next_return": next_return,
        "trend": trend, "mse": mse, "mae": mae, "rmse": rmse,
        "directional": directional, "pca": pca,
        "train_pca": train_pca, "y_test": y_test,
        "test_pred": test_pred, "days": days,
        "projection": projection, "k": k
    }

if run:
    if not ticker_text:
        st.error("Enter a stock ticker.")
    else:
        try:
            with st.spinner(f"Downloading and analyzing {ticker_text} data..."):
                data = get_stock_frame(ticker_text, period)
                result = analyze(data)

            st.subheader(f"{ticker_text} — Analysis results")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Latest closing price", f"${result['latest_price']:.2f}")
            c2.metric("Estimated next price", f"${result['estimated_price']:.2f}")
            c3.metric("Predicted trend", result["trend"])
            c4.metric("Directional accuracy", f"{result['directional']:.2f}%")

            st.caption(
                f"Estimated next-session return: {result['next_return'] * 100:.4f}%"
            )

            st.subheader("Model evaluation")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MAE", f"{result['mae']:.6f}")
            m2.metric("RMSE", f"{result['rmse']:.6f}")
            m3.metric("MSE", f"{result['mse']:.8f}")
            m4.metric(
                "PCA variance retained",
                f"{sum(result['pca'].explained_variance_ratio_) * 100:.2f}%"
            )

            left, right = st.columns(2)
            with left:
                st.markdown("#### Historical stock price")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(result["close"])
                ax.set_ylabel("Price (USD)")
                ax.grid(True)
                st.pyplot(fig)
                plt.close(fig)

            with right:
                st.markdown("#### PCA feature representation")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.scatter(
                    result["train_pca"][:, 0],
                    result["train_pca"][:, 1],
                    alpha=0.5
                )
                ax.set_xlabel("Principal Component 1")
                ax.set_ylabel("Principal Component 2")
                ax.grid(True)
                st.pyplot(fig)
                plt.close(fig)

            left, right = st.columns(2)
            with left:
                st.markdown("#### Actual vs predicted returns")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(result["y_test"] * 100, label="Actual")
                ax.plot(result["test_pred"] * 100, label="Predicted")
                ax.set_ylabel("Return (%)")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                plt.close(fig)

            with right:
                st.markdown("#### 30-day mathematical projection")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(result["days"], result["projection"])
                ax.set_xlabel("Days")
                ax.set_ylabel("Illustrative price (USD)")
                ax.grid(True)
                st.pyplot(fig)
                plt.close(fig)

            st.subheader("Mathematical methods used")
            st.markdown(r"""
            **1. Daily return**
            \[
            r_t = \frac{P_t-P_{t-1}}{P_{t-1}}
            \]

            **2. PCA and eigenvectors**
            \[
            Cv = \lambda v
            \]
            PCA reduces the five engineered features to two principal components.

            **3. Gradient descent**
            \[
            \theta_{\text{new}} = \theta_{\text{old}}-\alpha\nabla J(\theta)
            \]
            The model updates its weights to reduce mean squared error.

            **4. Differential equation**
            \[
            \frac{dP}{dt}=kP,\qquad P(t)=P_0e^{kt}
            \]
            This constant-growth curve is an illustration, not a reliable market forecast.
            """)

            st.warning(
                "Limitations: the model uses a simple linear predictor and a "
                "constant-growth projection. Historical test metrics do not guarantee "
                "future performance. This project is for educational purposes only."
            )

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
else:
    st.markdown("""
    ### Welcome
    Enter a stock ticker in the sidebar and select a historical period.

    Examples:
    - `AAPL` — Apple
    - `MSFT` — Microsoft
    - `TSLA` — Tesla
    - `RELIANCE.NS` — Reliance Industries (NSE)

    Then click **Analyze stock** to view the mathematical analysis.
    """)
