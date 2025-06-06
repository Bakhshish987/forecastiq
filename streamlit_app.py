import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------------------
# Page Setup & Styling
# ----------------------------
st.set_page_config(page_title="ForecastIQ", layout="wide")

st.title("📈 ForecastIQ: Stock Price Predictor")
st.write("Use the sidebar to select a stock and generate forecasts.")


# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("Welcome to ForecastIQ!")
ticker = st.sidebar.text_input("Enter stock ticker (e.g., AAPL, TSLA, AMZN)", value="AAPL")
n_days = st.sidebar.slider("Days to forecast", min_value=30, max_value=180, value=60, step=15)

with st.sidebar.expander("Popular Tickers Reference"):
    st.markdown("""
    | Company        | Ticker |
    |----------------|--------|
    | Apple          | AAPL   |
    | Tesla          | TSLA   |
    | Amazon         | AMZN   |
    | Microsoft      | MSFT   |
    | Nvidia         | NVDA   |
    | Google (Alphabet) | GOOGL |
    | Meta (Facebook)| META   |
    | Netflix        | NFLX   |
    | Bitcoin ETF    | BITO   |
    | S&P 500 ETF    | SPY    |
    | NASDAQ-100 ETF | QQQ    |
    | Dow Jones ETF  | DIA    |
    """)


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
        recent_price = df['y'].iloc[-1]  # Last known actual closing price

        # ----------------------------
        # Prophet Forecast
        # ----------------------------
        model = Prophet(daily_seasonality=True)
        model.fit(df)
        future = model.make_future_dataframe(periods=n_days)
        forecast = model.predict(future)

        # ----------------------------
        # Prediction Display (Big Font)
        # ----------------------------
        future_price = forecast.iloc[-1]['yhat']
        recent_avg = df['y'].tail(7).mean()
        
        # ----------------------------
        # Buy/Sell Recommendation
        # ----------------------------
        if future_price > recent_avg * 1.03:
            signal = "💰 BUY"
        elif future_price < recent_avg * 0.97:
            signal = "📉 SELL"
        else:
            signal = "HOLD"
                
        st.markdown(f"""
        <div style='text-align:center; font-size:24px; margin-bottom:25px;'>
            <p style='color:#555555; font-weight:500; margin:0;'>
                Most Recent Closing Price: ${recent_price:.2f}
            </p>
            <p style='margin:0;'>
                {n_days}-Day Forecasted Price:
                <span style='color:#3498DB; font-weight:bold;'>${future_price:.2f}</span>
            </p>
            <p style='margin:0;'>
                Action Recommendation: <strong>{signal}</strong>
            </p>
            <p style='font-size:18px; margin:0;'>
                Recent 7-Day Avg: ${recent_avg:.2f}
            </p>
        </div>
        """, unsafe_allow_html=True)





        # ----------------------------
        # Full-Range Forecast Plot (original)
        # ----------------------------
        st.subheader("📉 Full Forecast Overview (with historical data)")
        fig_full, ax_full = plt.subplots(figsize=(12, 6))
        ax_full.plot(df['ds'], df['y'], 'k.', label='⚫ Historical Price')
        ax_full.plot(forecast['ds'], forecast['yhat'], 'b-', label='🔵 Forecast')
        ax_full.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='skyblue', alpha=0.3, label='🔷 Confidence Interval')
        ax_full.set_title(f"{ticker} Full Stock Price Forecast", fontsize=16, fontweight='bold')
        ax_full.set_xlabel("Date")
        ax_full.set_ylabel("Price (USD)")
        ax_full.grid(True)
        ax_full.legend()
        st.pyplot(fig_full)

        # ----------------------------
        # Zoomed-In Forecast Plot
        # ----------------------------
        st.subheader("🔍 Zoomed-In Forecast (Last 30 Days + Future)")
        focus_start = forecast['ds'].iloc[-(n_days + 30)]
        focus_data = forecast[forecast['ds'] >= focus_start]

        fig_zoom, ax_zoom = plt.subplots(figsize=(12, 6))
        ax_zoom.plot(focus_data['ds'], focus_data['yhat'], label='🔵 Forecast', color='blue', linewidth=2)
        ax_zoom.fill_between(focus_data['ds'], focus_data['yhat_lower'], focus_data['yhat_upper'], color='skyblue', alpha=0.3)
        ax_zoom.set_title(f"{ticker} Focused Forecast (Next {n_days} Days)", fontsize=16, fontweight='bold')
        ax_zoom.set_xlabel("Date")
        ax_zoom.set_ylabel("Price (USD)")
        ax_zoom.grid(True)
        ax_zoom.legend()
        st.pyplot(fig_zoom)

        # ----------------------------
        # Forecast Components (Optional)
        # ----------------------------
        st.subheader("🧠 Prophet Model Components")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

        # ----------------------------
        # Model Evaluation
        # ----------------------------
        st.subheader("📈 Model Evaluation")
        merged = pd.merge(df, forecast[['ds', 'yhat']], on='ds', how='inner')
        mae = mean_absolute_error(merged['y'], merged['yhat'])
        rmse = mean_squared_error(merged['y'], merged['yhat']) ** 0.5

        col1, col2 = st.columns(2)
        with col1:
            st.metric("MAE", f"${mae:.2f}")
        with col2:
            st.metric("RMSE", f"${rmse:.2f}")

        # ----------------------------
        # Footer
        # ----------------------------
        st.markdown("""<hr style='margin-top: 50px;'>""", unsafe_allow_html=True)
        st.markdown("Built with ❤️ using [Streamlit](https://streamlit.io), [Prophet](https://facebook.github.io/prophet/), and [Yahoo Finance](https://finance.yahoo.com)", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**About ForecastIQ**

ForecastIQ is a stock price prediction application that uses live market data from Yahoo Finance and time series modeling with Facebook Prophet to forecast future price trends.
""")

