import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import time
import requests
from datetime import datetime, timedelta

# ----------------------------
# Page Setup & Styling
# ----------------------------
st.set_page_config(page_title="ForecastIQ", layout="wide")

st.title("📈 ForecastIQ: Real-Time Stock Forecasting")
st.write("Use the sidebar to select a stock and generate forecasts.")

# ----------------------------
# Helper Functions
# ----------------------------
def download_stock_data(ticker, start_date="2020-01-01", max_retries=3):
    """
    Enhanced stock data download with multiple fallback methods
    """
    for attempt in range(max_retries):
        try:
            # Method 1: Standard yfinance download
            data = yf.download(ticker, start=start_date, progress=False)
            
            if not data.empty and len(data) > 100:  # Ensure we have sufficient data
                return data
            
            # Method 2: Using Ticker object with different parameters
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, auto_adjust=True, back_adjust=True)
            
            if not data.empty and len(data) > 100:
                return data
            
            # Method 3: Try with different time periods
            data = stock.history(period="2y", auto_adjust=True)
            
            if not data.empty and len(data) > 100:
                return data
                
        except Exception as e:
            st.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
            continue
    
    return pd.DataFrame()  # Return empty DataFrame if all methods fail

def validate_ticker(ticker):
    """
    Validate if ticker exists and has recent data
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return 'symbol' in info or 'shortName' in info
    except:
        return False

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("Welcome to ForecastIQ!")
ticker = st.sidebar.text_input("Enter stock ticker (e.g., AAPL, TSLA, AMZN)", value="AAPL")
n_days = st.sidebar.slider("Days to forecast", min_value=30, max_value=180, value=60, step=15)

# Add data source options
data_period = st.sidebar.selectbox(
    "Data Period",
    ["2y", "1y", "6mo", "3mo", "1mo"],
    index=0
)

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

# Add troubleshooting info
with st.sidebar.expander("🔧 Troubleshooting"):
    st.markdown("""
    **If you're getting limited data:**
    - Try different data periods (6mo, 1y, 2y)
    - Check if ticker symbol is correct
    - Some tickers may have temporary issues
    - Wait a few minutes and try again
    """)

if st.sidebar.button("Run Forecast"):
    with st.spinner(f"Loading data for {ticker}..."):
        try:
            # Validate ticker first
            if not validate_ticker(ticker):
                st.error(f"❌ Invalid ticker symbol: {ticker}")
                st.stop()
            
            # Try multiple download methods
            st.info("🔄 Attempting to download stock data...")
            
            # Method 1: Using period parameter
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period=data_period, auto_adjust=True, back_adjust=True)
                
                if data.empty or len(data) < 50:
                    # Method 2: Using date range
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=730)  # 2 years
                    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if data.empty or len(data) < 50:
                    # Method 3: Alternative approach
                    data = download_stock_data(ticker, start_date="2022-01-01")
                
            except Exception as e:
                st.error(f"Error downloading data: {str(e)}")
                data = pd.DataFrame()
            
            # Check if we have sufficient data
            if data.empty:
                st.error(f"❌ No data found for {ticker}. Please check the ticker symbol.")
                st.stop()
            
            if len(data) < 50:
                st.error(f"❌ Insufficient data for {ticker}. Only {len(data)} days of data available.")
                st.stop()
            
            # Display data info
            st.success(f"✅ Successfully loaded {len(data)} days of data for {ticker}")
            st.info(f"📅 Date range: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
            
            # ----------------------------
            # Load & Prepare Data
            # ----------------------------
            df = data.reset_index()[['Date', 'Close']]
            df.columns = ['ds', 'y']
            df = df.dropna()
            df['ds'] = pd.to_datetime(df['ds'])
            # Remove timezone information if present (Prophet doesn't support timezones)
            if df['ds'].dt.tz is not None:
                df['ds'] = df['ds'].dt.tz_localize(None)
            df['y'] = df['y'].astype(float)
            recent_price = df['y'].iloc[-1]  # Last known actual closing price

            # --- RSI Calculation (Pandas only) ---
            delta = df['y'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14, min_periods=14).mean()
            avg_loss = loss.rolling(window=14, min_periods=14).mean()
            
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            latest_rsi = df['rsi'].iloc[-1]

            # ----------------------------
            # Prophet Forecast
            # ----------------------------
            with st.spinner("🔮 Generating forecast..."):
                model = Prophet(
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=True,
                    seasonality_mode='multiplicative'
                )
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
                signal = "⚖️ HOLD"
                    
            st.markdown(f"""
            <div style='text-align:center; font-size:24px; margin-bottom:25px;'>
                <p style='margin:0;'>
                    {ticker.upper()}
                </p>
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
                <p style='font-size:16px; color:#666; margin:0;'>
                    Current RSI: {latest_rsi:.1f}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # ----------------------------
            # Full-Range Forecast Plot (original)
            # ----------------------------
            st.subheader("📉 Full Forecast Overview (with historical data)")
            fig_full, ax_full = plt.subplots(figsize=(12, 6))
            ax_full.plot(df['ds'], df['y'], 'k.', label='⚫ Historical Price', markersize=1)
            ax_full.plot(forecast['ds'], forecast['yhat'], 'b-', label='🔵 Forecast', linewidth=2)
            ax_full.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='skyblue', alpha=0.3, label='🔷 Confidence Interval')
            ax_full.set_title(f"{ticker.upper()} Full Stock Price Forecast", fontsize=16, fontweight='bold')
            ax_full.set_xlabel("Date")
            ax_full.set_ylabel("Price (USD)")
            ax_full.grid(True, alpha=0.3)
            ax_full.legend()
            st.pyplot(fig_full)

            # ----------------------------
            # Zoomed-In Forecast Plot
            # ----------------------------
            st.subheader("🔍 Zoomed-In Forecast (Last 30 Days + Future)")
            focus_start = forecast['ds'].iloc[-(n_days + 30)]
            focus_data = forecast[forecast['ds'] >= focus_start]
            hist_data = df[df['ds'] >= focus_start]

            fig_zoom, ax_zoom = plt.subplots(figsize=(12, 6))
            ax_zoom.plot(hist_data['ds'], hist_data['y'], 'k.', label='⚫ Historical Price', markersize=2)
            ax_zoom.plot(focus_data['ds'], focus_data['yhat'], label='🔵 Forecast', color='blue', linewidth=2)
            ax_zoom.fill_between(focus_data['ds'], focus_data['yhat_lower'], focus_data['yhat_upper'], color='skyblue', alpha=0.3, label='🔷 Confidence Interval')
            ax_zoom.set_title(f"{ticker.upper()} Focused Forecast (Next {n_days} Days)", fontsize=16, fontweight='bold')
            ax_zoom.set_xlabel("Date")
            ax_zoom.set_ylabel("Price (USD)")
            ax_zoom.grid(True, alpha=0.3)
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
            mape = (abs(merged['y'] - merged['yhat']) / merged['y']).mean() * 100

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAE", f"${mae:.2f}")
            with col2:
                st.metric("RMSE", f"${rmse:.2f}")
            with col3:
                st.metric("MAPE", f"{mape:.1f}%")

            # ----------------------------
            # Trend Analysis: Moving Averages
            # ----------------------------
            st.subheader("📊 Trend Analysis: Moving Averages")
            
            df['MA_7'] = df['y'].rolling(window=7).mean()
            df['MA_30'] = df['y'].rolling(window=30).mean()
            
            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual Price', line=dict(color='black')))
            fig_ma.add_trace(go.Scatter(x=df['ds'], y=df['MA_7'], mode='lines', name='7-Day MA', line=dict(dash='dash', color='red')))
            fig_ma.add_trace(go.Scatter(x=df['ds'], y=df['MA_30'], mode='lines', name='30-Day MA', line=dict(dash='dot', color='green')))
            
            fig_ma.update_layout(title='Price Trend with Moving Averages', xaxis_title='Date', yaxis_title='Price')
            st.plotly_chart(fig_ma, use_container_width=True)

            # ----------------------------
            # RSI Visualization
            # ----------------------------
            st.subheader("📉 RSI (Relative Strength Index)")
            
            fig_rsi = go.Figure()
            
            # Plot RSI line
            fig_rsi.add_trace(go.Scatter(
                x=df['ds'],
                y=df['rsi'],
                mode='lines',
                name='RSI',
                line=dict(color='orange')
            ))
            
            # Add overbought (70) and oversold (30) lines
            fig_rsi.add_shape(
                type='line',
                x0=df['ds'].min(),
                x1=df['ds'].max(),
                y0=70,
                y1=70,
                line=dict(color='red', dash='dash')
            )
            
            fig_rsi.add_shape(
                type='line',
                x0=df['ds'].min(),
                x1=df['ds'].max(),
                y0=30,
                y1=30,
                line=dict(color='green', dash='dash')
            )
            
            # Add annotations
            fig_rsi.add_annotation(x=df['ds'].iloc[-1], y=70, text="Overbought", showarrow=False, yshift=10)
            fig_rsi.add_annotation(x=df['ds'].iloc[-1], y=30, text="Oversold", showarrow=False, yshift=-10)
            
            # Layout tweaks
            fig_rsi.update_layout(
                title='14-Day RSI Indicator',
                yaxis_title='RSI Value',
                xaxis_title='Date',
                showlegend=False,
                height=400
            )
            
            # Display chart
            st.plotly_chart(fig_rsi, use_container_width=True)

            # ----------------------------
            # Forecast Residuals
            # ----------------------------
            st.subheader("📉 Forecast Error Distribution")
            
            merged['residual'] = merged['y'] - merged['yhat']
            
            fig_resid = go.Figure()
            fig_resid.add_trace(go.Histogram(x=merged['residual'], nbinsx=50, marker_color='indianred'))
            
            fig_resid.update_layout(title='Residuals: Actual - Forecast', xaxis_title='Error', yaxis_title='Frequency')
            st.plotly_chart(fig_resid, use_container_width=True)

            # ----------------------------
            # Download Forecast
            # ----------------------------
            st.subheader("📥 Download Forecast Data")
            
            download_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            download_df.columns = ['Date', 'Forecast', 'Lower_Bound', 'Upper_Bound']
            download_csv = download_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="Download Forecast CSV",
                data=download_csv,
                file_name=f"{ticker}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv'
            )

            # ----------------------------
            # Footer
            # ----------------------------
            st.markdown("""<hr style='margin-top: 50px;'>""", unsafe_allow_html=True)
            st.markdown("Built with ❤️ using [Streamlit](https://streamlit.io), [Prophet](https://facebook.github.io/prophet/), and [Yahoo Finance](https://finance.yahoo.com)", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Try the following:")
            st.markdown("""
            - Check if the ticker symbol is correct
            - Try a different data period (6mo, 1y, 2y)
            - Wait a few minutes and try again
            - Some tickers may have temporary issues with Yahoo Finance
            """)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**About ForecastIQ**

ForecastIQ is a stock price prediction application that uses live market data from Yahoo Finance and time series modeling with Facebook Prophet to forecast future price trends.

**Recent Updates:**
- Enhanced data fetching with multiple fallback methods
- Better error handling and user feedback
- Improved data validation

Built by Bakhshish Sethi.
""")
