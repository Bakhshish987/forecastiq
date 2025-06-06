import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="ForecastIQ", layout="wide")

st.title("📈 ForecastIQ: Stock Price Predictor")
st.write("Predict stock prices using Facebook Prophet and live Yahoo Finance data.")

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("User Input")

ticker = st.sidebar.text_input("Enter stock ticker (e.g., AAPL, TSLA, AMZN)", value="AAPL")
n_days = st.sidebar.slider("Days to forecast", min_value=30, max_value=180, value=60, step=15)

if st.sidebar.button("Run Forecast"):
    # ----------------------------
    # Load & Prepare Data
    # ----------------------------
    try:
        data = yf.download(ticker, start="2019-01-01")
        df = data.reset_index()[['Date', 'Close']]
        df.columns = ['ds', 'y']
        df = df.dropna()
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = df['y'].astype(float)

        # ----------------------------
        # Prophet Forecast
        # ----------------------------
        model = Prophet(daily_seasonality=True)
        model.fit(df)
        future = model.make_future_dataframe(periods=n_days)
        forecast = model.predict(future)

        # ----------------------------
        # Plot Forecast
        # ----------------------------
        st.subheader(f"{ticker} Stock Price Forecast ({n_days} days)")
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        # ----------------------------
        # Plot Components (Trend, Seasonality)
        # ----------------------------
        st.subheader("Forecast Components")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

        # ----------------------------
        # Metrics & Buy/Sell Signal
        # ----------------------------
        st.subheader("📊 Model Evaluation")
        merged = pd.merge(df, forecast[['ds', 'yhat']], on='ds', how='inner')
        mae = mean_absolute_error(merged['y'], merged['yhat'])
        rmse = mean_squared_error(merged['y'], merged['yhat']) ** 0.5

        st.write(f"**MAE:** ${mae:.2f}")
        st.write(f"**RMSE:** ${rmse:.2f}")

        # ----------------------------
        # Signal Logic
        # ----------------------------
        future_price = forecast.iloc[-1]['yhat']
        recent_avg = df['y'].tail(7).mean()

        if future_price > recent_avg * 1.03:
            signal = "💰 BUY"
        elif future_price < recent_avg * 0.97:
            signal = "📉 SELL"
        else:
            signal = "🤝 HOLD"

        st.subheader("📌 Recommendation")
        st.write(f"**Predicted price:** ${future_price:.2f}")
        st.write(f"**Recent avg (7d):** ${recent_avg:.2f}")
        st.markdown(f"### 🔥 ACTION: **{signal}**")

    except Exception as e:
        st.error(f"Error: {e}")
