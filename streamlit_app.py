import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------------------
# Page Setup & Branding
# ----------------------------
st.set_page_config(page_title="ForecastIQ", layout="wide")

st.markdown("""
    <style>
        .main-title {
            font-size:42px;
            font-weight:bold;
            color:#0E1117;
        }
        .subtitle {
            font-size:18px;
            color:#6c757d;
        }
        .big-number {
            font-size:36px;
            font-weight:bold;
            color:#0E76A8;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>📈 ForecastIQ: Stock Price Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Powered by Yahoo Finance & Facebook Prophet • Built by Bakhshish Sethi</div><br>", unsafe_allow_html=True)

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Enter stock ticker (e.g., AAPL, TSLA, AMZN)", value="AAPL")
n_days = st.sidebar.slider("Days to forecast", min_value=30, max_value=180, value=60, step=15)

if st.sidebar.button("Run Forecast"):
    try:
        # ----------------------------
        # Load & Prepare Data
        # ----------------------------
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
        # Forecasted Price Display
        # ----------------------------
        future_price = forecast.iloc[-1]['yhat']
        st.markdown(f"""
            <h2 style='text-align: center;'>💰 Predicted Price: <span style='color:#0E76A8;'>${future_price:.2f}</span></h2>
        """, unsafe_allow_html=True)

        # ----------------------------
        # Zoomed Forecast Plot
        # ----------------------------
        focus_start = forecast['ds'].iloc[-(n_days + 30)]
        focus_data = forecast[forecast['ds'] >= focus_start]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(focus_data['ds'], focus_data['yhat'], label='🔵 Forecast', color='blue', linewidth=2)
        ax.fill_between(focus_data['ds'], focus_data['yhat_lower'], focus_data['yhat_upper'], color='skyblue', alpha=0.3)
        ax.set_title(f"{ticker} Stock Price Forecast (Last 30 + Next {n_days} Days)", fontsize=16, fontweight='bold')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        # ----------------------------
        # Forecast Components
        # ----------------------------
        st.subheader("📊 Forecast Components (Trend, Seasonality)")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

        # ----------------------------
        # Metrics & Buy/Sell Signal
        # ----------------------------
        st.subheader("📈 Model Evaluation & Recommendation")
        merged = pd.merge(df, forecast[['ds', 'yhat']], on='ds', how='inner')
        mae = mean_absolute_error(merged['y'], merged['yhat'])
        rmse = mean_squared_error(merged['y'], merged['yhat']) ** 0.5
        recent_avg = df['y'].tail(7).mean()

        # Signal logic
        if future_price > recent_avg * 1.03:
            signal = "💰 BUY"
        elif future_price < recent_avg * 0.97:
            signal = "📉 SELL"
        else:
            signal = "🤝 HOLD"

        # Columns Layout
        col1, col2 = st.columns(2)
        with col1:
            st.metric("MAE", f"${mae:.2f}")
            st.metric("RMSE", f"${rmse:.2f}")
        with col2:
            st.markdown("### 🔥 Action Recommendation")
            st.markdown(f"**Recent Avg (7d):** ${recent_avg:.2f}")
            st.markdown(f"### ✅ Recommended: **{signal}**")

        # ----------------------------
        # Footer
        # ----------------------------
        st.markdown("""<hr style='margin-top: 50px;'>""", unsafe_allow_html=True)
        st.markdown("Built with ❤️ using [Streamlit](https://streamlit.io), [Prophet](https://facebook.github.io/prophet/), and [Yahoo Finance](https://finance.yahoo.com)", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")

