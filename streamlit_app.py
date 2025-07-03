import streamlit as st
import pandas as pd
import requests
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go

# Config
st.set_page_config(page_title="ForecastIQ", layout="wide")
API_KEY = "IYL3O9IG01MW9ZT1"

st.title("📈 ForecastIQ: Real-Time Stock Forecasting")
st.write("Use the sidebar to select a stock and generate forecasts.")

# Sidebar
st.sidebar.header("Welcome to ForecastIQ!")
ticker = st.sidebar.text_input("Enter stock ticker (e.g., AAPL, TSLA)", value="AAPL")
n_days = st.sidebar.slider("Days to forecast", min_value=30, max_value=180, value=60, step=15)

if st.sidebar.button("Run Forecast"):
    try:
        # Fetch data from Alpha Vantage
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={API_KEY}"
        response = requests.get(url)
        data = response.json()

        if "Time Series (Daily)" not in data:
            st.error("❌ Could not fetch data. Check ticker or try again later.")
            st.stop()

        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
        df = df.rename(columns={"5. adjusted close": "y"})
        df["ds"] = pd.to_datetime(df.index)
        df["y"] = df["y"].astype(float)
        df = df[["ds", "y"]].sort_values("ds").reset_index(drop=True)
        df['ds'] = pd.to_datetime(df['ds'])

        st.write("✅ Latest 5 rows of stock data:")
        st.dataframe(df.tail())

        recent_price = df['y'].iloc[-1]
        recent_avg = df['y'].tail(7).mean()

        # RSI Calculation
        delta = df['y'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        latest_rsi = df['rsi'].iloc[-1]

        # Forecasting
        model = Prophet(daily_seasonality=True)
        model.fit(df)
        future = model.make_future_dataframe(periods=n_days)
        forecast = model.predict(future)
        future_price = forecast.iloc[-1]['yhat']

        # Recommendation
        if future_price > recent_avg * 1.03:
            signal = "💰 BUY"
        elif future_price < recent_avg * 0.97:
            signal = "📉 SELL"
        else:
            signal = "HOLD"

        st.markdown(f"""
        <div style='text-align:center; font-size:24px; margin-bottom:25px;'>
            <p>{ticker}</p>
            <p style='color:#555;'>Recent Close: ${recent_price:.2f}</p>
            <p>{n_days}-Day Forecast: <span style='color:#3498DB;'>${future_price:.2f}</span></p>
            <p>Action Recommendation: <strong>{signal}</strong></p>
            <p style='font-size:18px;'>Recent 7-Day Avg: ${recent_avg:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

        # Full Forecast
        st.subheader("📉 Full Forecast Overview")
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df['ds'], df['y'], 'k.', label='Actual')
        ax1.plot(forecast['ds'], forecast['yhat'], 'b-', label='Forecast')
        ax1.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='skyblue', alpha=0.3)
        ax1.legend()
        ax1.grid(True)
        st.pyplot(fig1)

        # Zoomed Forecast
        st.subheader("🔍 Zoomed-In Forecast")
        focus_start = forecast['ds'].iloc[-(n_days + 30)]
        focus_data = forecast[forecast['ds'] >= focus_start]
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(focus_data['ds'], focus_data['yhat'], color='blue', label='Forecast')
        ax2.fill_between(focus_data['ds'], focus_data['yhat_lower'], focus_data['yhat_upper'], color='skyblue', alpha=0.3)
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)

        # Model Components
        st.subheader("🧠 Prophet Model Components")
        fig3 = model.plot_components(forecast)
        st.pyplot(fig3)

        # Evaluation
        st.subheader("📈 Model Evaluation")
        merged = pd.merge(df, forecast[['ds', 'yhat']], on='ds', how='inner')
        mae = mean_absolute_error(merged['y'], merged['yhat'])
        rmse = mean_squared_error(merged['y'], merged['yhat']) ** 0.5
        col1, col2 = st.columns(2)
        col1.metric("MAE", f"${mae:.2f}")
        col2.metric("RMSE", f"${rmse:.2f}")

        # Moving Averages
        st.subheader("📊 Trend: Moving Averages")
        df['MA_7'] = df['y'].rolling(7).mean()
        df['MA_30'] = df['y'].rolling(30).mean()
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Price'))
        fig_ma.add_trace(go.Scatter(x=df['ds'], y=df['MA_7'], name='7-day MA'))
        fig_ma.add_trace(go.Scatter(x=df['ds'], y=df['MA_30'], name='30-day MA'))
        fig_ma.update_layout(title='Moving Averages', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_ma)

        # RSI Chart
        st.subheader("📉 RSI (14-day)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df['ds'], y=df['rsi'], name='RSI', line=dict(color='orange')))
        fig_rsi.add_shape(type='line', x0=df['ds'].min(), x1=df['ds'].max(), y0=70, y1=70, line=dict(color='red', dash='dash'))
        fig_rsi.add_shape(type='line', x0=df['ds'].min(), x1=df['ds'].max(), y0=30, y1=30, line=dict(color='green', dash='dash'))
        fig_rsi.update_layout(title='RSI Indicator', yaxis_title='RSI', xaxis_title='Date')
        st.plotly_chart(fig_rsi)

        # Residuals
        st.subheader("📉 Forecast Residuals")
        merged['residual'] = merged['y'] - merged['yhat']
        fig_res = go.Figure()
        fig_res.add_trace(go.Histogram(x=merged['residual'], nbinsx=50, marker_color='indianred'))
        fig_res.update_layout(title='Residual Distribution', xaxis_title='Error', yaxis_title='Frequency')
        st.plotly_chart(fig_res)

        # Download CSV
        st.subheader("📥 Download Forecast")
        download_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        st.download_button(
            label="Download Forecast CSV",
            data=download_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_forecast.csv",
            mime="text/csv"
        )

        st.markdown("---")
        st.markdown("Built with ❤️ by **Bakhshish Sethi**")

    except Exception as e:
        st.error(f"❌ Error: {e}")
