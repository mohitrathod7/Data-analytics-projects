# %%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==========================================
# Utility and Helper Functions
# ==========================================
def evaluate_model(true, pred):
    """Return a dictionary of evaluation metrics given true and predicted values."""
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true, pred)
    mape = np.mean(np.abs((true - pred) / true)) * 100
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'MSE': mse}

def plot_time_series_chart(df, x_col, y_col, y_label='Y Axis', title='Line Plot'):
    """Plot a single line over time."""
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df[x_col], df[y_col], color='green', label=y_col)
    plt.xlabel(x_col.capitalize())
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return fig

def detect_outliers_iqr(df, column, date_column='Date'):
    """
    Detect and visualize outliers in a specified column using the IQR method.
    Returns a figure and a sample of the outliers.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df[date_column], df[column], label=column.capitalize(), color='green')
    plt.scatter(outliers[date_column], outliers[column], color='red', label='Outliers')
    plt.xlabel('Date')
    plt.ylabel(column.capitalize())
    plt.title(f'{column.capitalize()} with Outliers')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    return fig, outliers.head()

def adf_test(series, column_name='Time Series'):
    """Run the Augmented Dickey-Fuller test and return the result."""
    result = adfuller(series.dropna(), autolag='AIC')
    return result

# ==========================================
# MODEL-SPECIFIC FORECASTING FUNCTIONS
# ==========================================
def run_arima(train, test, forecast_index):
    """Run ARIMA model, return (metrics_dict, figure)."""
    from statsmodels.tsa.arima.model import ARIMA
    fig = None
    metrics = {}
    try:
        arima_model = ARIMA(train, order=(7, 1, 9))
        arima_fit = arima_model.fit()
        arima_pred = arima_fit.predict(start=len(train), end=len(train)+len(test)-1)
        arima_pred.index = forecast_index
        
        metrics = evaluate_model(test['Close'], arima_pred)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(test.index, test['Close'], label='Actual', linewidth=2, color='green')
        ax.plot(test.index, arima_pred, color='red', linestyle='--', label='ARIMA Forecast')
        ax.set_title("ARIMA: Actual vs Predicted")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
    except Exception as e:
        st.error(f"Error in ARIMA: {e}")
    return metrics, fig

def run_ets(train, test, forecast_index):
    """Run ETS (SES) model, return (metrics_dict, figure)."""
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    fig = None
    metrics = {}
    try:
        ses_model = SimpleExpSmoothing(train['Close']).fit(smoothing_level=0.2)
        ses_pred = ses_model.predict(start=test.index[0], end=test.index[-1])
        metrics = evaluate_model(test['Close'], ses_pred)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train.index, train['Close'], label='Train (Actual)', color='blue', alpha=0.5)
        ax.plot(test.index, test['Close'], label='Test (Actual)', color='green', linewidth=2)
        ax.plot(test.index, ses_pred, label='ETS Forecast', color='red', linestyle='--')
        ax.set_title("ETS: Actual vs Predicted")
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
    except Exception as e:
        st.error(f"Error in ETS: {e}")
    return metrics, fig

def run_prophet(train, test, forecast_index):
    """Run Prophet model, return (metrics_dict, figure)."""
    fig = None
    metrics = {}
    try:
        from holidays import Canada
        from prophet import Prophet
        
        train_prophet = train.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
        test_prophet = test.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
        
        # Using Canadian holidays as an example
        canada_holidays = Canada(years=range(train_prophet['ds'].min().year, test_prophet['ds'].max().year + 2))
        holiday_df = pd.DataFrame([{'ds': date, 'holiday': 'canadian_holiday'} for date in canada_holidays])
        
        model = Prophet(
            holidays=holiday_df,
            daily_seasonality=False,
            yearly_seasonality=True,
            weekly_seasonality=False
        )
        model.fit(train_prophet)
        
        future = model.make_future_dataframe(periods=len(test), freq='M')
        forecast = model.predict(future)
        
        forecast_filtered = forecast.set_index('ds').loc[forecast_index]['yhat']
        metrics = evaluate_model(test['Close'], forecast_filtered)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(test.index, test['Close'], label='Actual', linewidth=2, color='green')
        ax.plot(test.index, forecast_filtered, color='blue', linestyle='--', label='Prophet Forecast')
        ax.set_title("Prophet: Actual vs Predicted")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
    except Exception as e:
        st.error(f"Error in Prophet: {e}")
    return metrics, fig

def run_lstm(train, test, monthly_price, split_index):
    """Run LSTM model, return (metrics_dict, figure)."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from sklearn.preprocessing import MinMaxScaler
    
    fig = None
    metrics = {}
    try:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(monthly_price.values.reshape(-1, 1))

        def create_lagged_dataset(data, lag=12):
            X, y = [], []
            for i in range(lag, len(data)):
                X.append(data[i-lag:i])
                y.append(data[i])
            return np.array(X), np.array(y)

        lag = 12
        X_data, y_data = create_lagged_dataset(scaled, lag)
        train_X, test_X = X_data[:split_index - lag], X_data[split_index - lag:]
        train_y, test_y = y_data[:split_index - lag], y_data[split_index - lag:]
        
        train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], 1))
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], 1))

        lstm_model = Sequential()
        lstm_model.add(LSTM(50, activation='relu', input_shape=(lag, 1)))
        lstm_model.add(Dense(1))
        lstm_model.compile(optimizer='adam', loss='mse')
        lstm_model.fit(train_X, train_y, epochs=100, verbose=0)

        lstm_pred = lstm_model.predict(test_X)
        lstm_pred = scaler.inverse_transform(lstm_pred)
        actual = scaler.inverse_transform(test_y.reshape(-1, 1)).flatten()

        metrics = evaluate_model(actual, lstm_pred.flatten())
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(test.index, test['Close'], label='Actual', linewidth=2, color='green')
        ax.plot(test.index, lstm_pred.flatten(), color='red', linestyle='--', label='LSTM Forecast')
        ax.set_title("LSTM: Actual vs Predicted")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
    except Exception as e:
        st.error(f"Error in LSTM: {e}")
    return metrics, fig

# ==========================================
# STREAMLIT APP START
# ==========================================
st.title("Stock Price Decomposition, Diagnostics & Forecasting")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # ------------------------------
    # 1) Load and Basic Checks
    # ------------------------------
    df = pd.read_csv(uploaded_file)
    if 'Close' not in df.columns:
        st.error("The uploaded CSV must contain a 'Close' column.")
        st.stop()

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values(by='Date')
        fig_ts = plot_time_series_chart(df, 'Date', 'Close', y_label='Close Price', title='Stock: Close Price Over Time')
        st.header("Time Series Plot")
        st.pyplot(fig_ts)
    else:
        st.header("Time Series Plot")
        st.line_chart(df['Close'])

    # ------------------------------
    # 2) Outlier Detection
    # ------------------------------
    st.header("Outlier Detection")
    fig_outliers, outliers_sample = detect_outliers_iqr(df, column='Close', date_column='Date')
    st.pyplot(fig_outliers)
    st.write(outliers_sample)

    # ------------------------------
    # 3) Decomposition: Select Method
    # ------------------------------
    st.header("Time Series Decomposition")
    decomp_choice = st.selectbox("Choose Decomposition Method", ["Additive", "Multiplicative"])
    try:
        if decomp_choice == "Additive":
            decomp_result = seasonal_decompose(df['Close'], model='additive', period=30)
        else:
            decomp_result = seasonal_decompose(df['Close'], model='multiplicative', period=30)
        
        fig_decomp, axs = plt.subplots(4, 1, figsize=(14, 10))
        axs[0].plot(df['Date'], df['Close'], color='green')
        axs[0].set_title('Original')
        axs[1].plot(df['Date'], decomp_result.trend, color='green')
        axs[1].set_title('Trend')
        axs[2].plot(df['Date'], decomp_result.seasonal, color='green')
        axs[2].set_title('Seasonal')
        axs[3].plot(df['Date'], decomp_result.resid, color='green')
        axs[3].set_title('Residual')
        plt.tight_layout()
        st.pyplot(fig_decomp)
    except Exception as e:
        st.error(f"Error in {decomp_choice} decomposition: {e}")

    # ------------------------------
    # 4) Stationarity Tests
    # ------------------------------
    st.header("Stationarity Test (ADF) on Original Series")
    adf_result_original = adf_test(df['Close'], 'Close Price')
    labels = ['ADF Statistic', 'p-value', '# Lags Used', '# Observations Used']
    for val, label in zip(adf_result_original[:4], labels):
        st.write(f"{label}: {val}")
    if adf_result_original[1] <= 0.05:
        st.write("✅ The series is stationary (reject null hypothesis)")
    else:
        st.write("❌ The series is non-stationary (fail to reject null hypothesis)")

    st.header("First Order Differencing & ACF Plot")
    df['Close_Diff'] = df['Close'].diff()
    diff_series = df['Close_Diff'].dropna()
    adf_result_diff = adf_test(diff_series, 'Close Price (1st Difference)')
    st.write("ADF Test on the first differenced series:")
    for val, label in zip(adf_result_diff[:4], labels):
        st.write(f"{label}: {val}")
    if adf_result_diff[1] <= 0.05:
        st.write("✅ The differenced series is stationary (reject null hypothesis)")
    else:
        st.write("❌ The differenced series is still non-stationary (fail to reject null hypothesis)")
    
    fig_acf = plt.figure(figsize=(12, 6))
    plot_acf(diff_series, ax=plt.gca(), lags=30)
    plt.title('ACF of Stock (1st Order Differenced)')
    plt.tight_layout()
    st.pyplot(fig_acf)

    # ------------------------------
    # 5) Simulated Series
    # ------------------------------
    st.header("Simulated Series Analysis: White Noise vs Random Walk")
    np.random.seed(0)
    n = 1000
    white_noise = np.random.normal(0, 1, n)
    random_walk = np.cumsum(np.random.normal(0, 1, n))

    fig_sim = plt.figure(figsize=(14, 6))
    plt.subplot(2, 1, 1)
    plt.plot(white_noise, color='green')
    plt.title('White Noise')
    plt.subplot(2, 1, 2)
    plt.plot(random_walk, color='green')
    plt.title('Random Walk')
    plt.tight_layout()
    st.pyplot(fig_sim)

    # ------------------------------
    # 6) Forecasting: Run All Models
    # ------------------------------
    st.header("Forecasting Evaluation")
    if df.index.name != 'Date':
        df = df.set_index('Date')

    monthly_price = df['Close'].resample('M').last()
    monthly_price = monthly_price.to_frame(name='Close')
    st.write("Monthly aggregated Close prices (sample):")
    st.write(monthly_price.head())

    split_index = int(len(monthly_price) * 0.8)
    train, test = monthly_price[:split_index], monthly_price[split_index:]
    forecast_index = test.index

    # Run all models in background
    arima_metrics, arima_fig = run_arima(train, test, forecast_index)
    ets_metrics, ets_fig = run_ets(train, test, forecast_index)
    prophet_metrics, prophet_fig = run_prophet(train, test, forecast_index)
    lstm_metrics, lstm_fig = run_lstm(train, test, monthly_price, split_index)

    # ------------------------------
    # 7) Display Selected Model Details
    # ------------------------------
    model_choice = st.selectbox("Select Forecasting Model Details to Display", 
                                ["ARIMA", "ETS", "Prophet", "LSTM"])

    if model_choice == "ARIMA":
        if arima_fig:
            st.subheader("Forecast Evaluation Metrics (ARIMA)")
            st.write(arima_metrics)
            st.pyplot(arima_fig)
    elif model_choice == "ETS":
        if ets_fig:
            st.subheader("Forecast Evaluation Metrics (ETS)")
            st.write(ets_metrics)
            st.pyplot(ets_fig)
    elif model_choice == "Prophet":
        if prophet_fig:
            st.subheader("Forecast Evaluation Metrics (Prophet)")
            st.write(prophet_metrics)
            st.pyplot(prophet_fig)
    elif model_choice == "LSTM":
        if lstm_fig:
            st.subheader("Forecast Evaluation Metrics (LSTM)")
            st.write(lstm_metrics)
            st.pyplot(lstm_fig)

    # ------------------------------
    # 8) Forecasting Metrics Comparison & Metric Select Box
    # ------------------------------
    st.header("Forecasting Metrics Comparison")
    forecast_comparison = pd.DataFrame({
        'ARIMA': arima_metrics,
        'ETS': ets_metrics,
        'Prophet': prophet_metrics,
        'LSTM': lstm_metrics
    }).T

    for col in forecast_comparison.select_dtypes(include=['number']):
        forecast_comparison[col] = forecast_comparison[col].round(2)

    st.dataframe(forecast_comparison)

    selected_metric = st.selectbox("Select Metric to Display", ["RMSE", "MAE", "MAPE", "MSE"])

    if selected_metric in forecast_comparison.columns:
        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
        models = forecast_comparison.index
        metric_values = forecast_comparison[selected_metric]
        ax_bar.bar(models, metric_values, color=['blue', 'orange', 'green', 'red'])
        ax_bar.set_xlabel("Forecasting Models")
        ax_bar.set_ylabel(selected_metric)
        ax_bar.set_title(f"{selected_metric} Comparison for Forecasting Models")
        plt.tight_layout()
        st.pyplot(fig_bar)
else:
    st.info("Please upload a CSV file to begin.")
