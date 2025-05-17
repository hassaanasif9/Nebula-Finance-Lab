import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import plotly.express as px
import plotly.graph_objects as go
import ta
import xgboost as xgb
import openpyxl
from datetime import datetime
import warnings
import sys
import traceback
import time

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Utility Functions
@st.cache_data(ttl=3600)
def download_stock_data(ticker, period, retries=3, backoff_factor=2):
    """Download stock data from Yahoo Finance with retry and fallback period"""
    try:
        ticker = ticker.strip().upper()
        if not ticker:
            st.error("🚫 Please enter a valid ticker symbol")
            return None

        # Define period mappings
        period_map = {
            "1mo": pd.Timedelta(days=35),
            "3mo": pd.Timedelta(days=95),
            "6mo": pd.Timedelta(days=185),
            "1y": pd.Timedelta(days=370),
            "2y": pd.Timedelta(days=740),
            "5y": pd.Timedelta(days=1850),
            "max": pd.Timedelta(days=3650)
        }
        end_date = datetime.now()
        start_date = end_date - period_map.get(period, pd.Timedelta(days=370))

        # List of periods to try (start with selected, fallback to longer periods)
        periods_to_try = [period]
        if period in ["1mo", "3mo"]:
            periods_to_try.append("1y")  # Fallback to 1y if short period fails

        for try_period in periods_to_try:
            st.write(f"Attempting to fetch data for {ticker} with period {try_period}...")
            start_date = end_date - period_map.get(try_period, pd.Timedelta(days=370))

            # Retry loop for the current period
            for attempt in range(1, retries + 1):
                try:
                    df = yf.download(
                        ticker,
                        start=start_date,
                        end=end_date,
                        progress=False,
                        auto_adjust=False,
                        prepost=False,
                        threads=False,
                        timeout=10
                    )
                    
                    # Debug information
                    st.write(f"Attempt {attempt}: Downloaded data shape: {df.shape}")
                    st.write(f"Columns: {df.columns.tolist()}")
                    if not df.empty:
                        st.write(f"Date range: {df.index.min()} to {df.index.max()}")

                    if df is not None and not df.empty:
                        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                        df = df.rename(columns={
                            'Adj Close': 'Close',
                            'Adj. Close': 'Close',
                            'AdjClose': 'Close'
                        })
                        missing_columns = [col for col in required_columns if col not in df.columns]
                        
                        if not missing_columns:
                            if not df['Close'].isna().all():
                                df['Ticker'] = ticker
                                st.success(f"✅ Successfully fetched data for {ticker} with period {try_period}")
                                return df
                            else:
                                st.warning(f"Attempt {attempt}: Close column contains only NaN values")
                        else:
                            st.warning(f"Attempt {attempt}: Missing columns: {missing_columns}")
                    
                    # Wait before retrying
                    if attempt < retries:
                        time.sleep(backoff_factor * attempt)
                        st.warning(f"Attempt {attempt} failed for period {try_period}, retrying...")
                    continue
                    
                except Exception as e:
                    st.warning(f"Attempt {attempt} failed for period {try_period}: {str(e)}")
                    if attempt < retries:
                        time.sleep(backoff_factor * attempt)
                        continue
                    else:
                        st.warning(f"All retries failed for period {try_period}")
                        break  # Move to next period if available

        # If all periods fail
        st.error(f"🚫 Failed to fetch data for {ticker}. Please try:")
        st.markdown("""
        - Selecting a longer period (e.g., '1y' or 'max')
        - Clearing the cache (click '🗑 Clear Cache')
        - Checking your internet connection
        - Trying a different ticker (e.g., MSFT, GOOGL)
        - Waiting a few minutes and retrying
        """)
        return None
        
    except Exception as e:
        error_msg = str(e).lower()
        if "jsondecodeerror" in error_msg or "expecting value" in error_msg:
            st.error("🚫 Network or API response error. Please check your connection and try again.")
        elif "symbol may be delisted" in error_msg:
            st.error(f"🚫 {ticker} may be delisted or unavailable.")
        else:
            st.error(f"🚫 Unexpected error: {str(e)}")
        return None

def standardize_column_names(df):
    """Standardize column names to match expected format"""
    column_mapping = {
        'Price': 'Close',
        'price': 'Close',
        'PRICE': 'Close',
        'Adj Price': 'Adj Close',
        'adj price': 'Adj Close',
        'ADJ PRICE': 'Adj Close'
    }
    df = df.copy()
    df.columns = [column_mapping.get(col, col) for col in df.columns]
    return df

@st.cache_data(ttl=3600)
def process_dataframe(df, feature):
    """Process dataframe with feature engineering"""
    try:
        if df is None or df.empty:
            return None
        df = standardize_column_names(df)
        df_processed = df.copy()
        
        if feature == "Returns":
            df_processed['Returns'] = df_processed['Close'].pct_change()
        elif feature == "RSI":
            if len(df_processed) >= 14:
                df_processed['RSI'] = ta.momentum.RSIIndicator(df_processed['Close'], window=14).rsi()
            else:
                st.error("⚠ Need at least 14 rows for RSI calculation")
                return None
        elif feature == "Moving Average":
            if len(df_processed) >= 20:
                df_processed['MA20'] = df_processed['Close'].rolling(window=20).mean()
            else:
                st.error("⚠ Need at least 20 rows for Moving Average calculation")
                return None
                
        df_processed = df_processed.dropna()
        if df_processed.empty:
            st.error("⚠ No valid data after processing")
            return None
        return df_processed
    except Exception as e:
        st.error(f"🚫 Error processing data: {str(e)}")
        return None

def validate_data(df, feature):
    """Validate dataframe for analysis"""
    if df is None or df.empty:
        return False, "Dataframe is empty or None."
    if not isinstance(df.index, pd.DatetimeIndex):
        return False, "Data must have a datetime index."
    if df.isnull().sum().sum() > len(df) * 0.5:
        return False, "Too many missing values."
    min_rows = 20 if feature == "Moving Average" else 14 if feature == "RSI" else 2
    if len(df) < min_rows:
        return False, f"Need at least {min_rows} rows for {feature}."
    if isinstance(df.columns, pd.MultiIndex):
        return False, "MultiIndex columns detected. Please flatten columns."
    price_columns = ['Close', 'Price']
    if not any(col in df.columns for col in price_columns):
        return False, "Missing required price column."
    return True, "Data is valid."

def flatten_multiindex_columns(df):
    """Flatten multiindex columns if they exist"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]
    return df

def check_dependencies():
    """Check for required packages"""
    missing_packages = []
    try:
        import xgboost
    except ImportError:
        missing_packages.append("xgboost")
    try:
        import openpyxl
    except ImportError:
        missing_packages.append("openpyxl")
    if missing_packages:
        st.error(f"🚫 Missing required packages: {', '.join(missing_packages)}")
        st.info("Please install using: pip install " + " ".join(missing_packages))
        st.stop()

def train_model(X, y, model_type, test_size=0.2):
    """Train machine learning model with error handling"""
    try:
        if X is None or y is None or X.empty or y.empty:
            st.error("🚫 Invalid input data for model training")
            return None
        if model_type == "K-Means Clustering":
            X_train, X_test = X.iloc[:-int(test_size*len(X))], X.iloc[-int(test_size*len(X)):]
            y_train, y_test = None, None
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=False, random_state=42
            )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if model_type == "Linear Regression":
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
        elif model_type == "Logistic Regression":
            y_train = (y_train > 0).astype(int)
            y_test = (y_test > 0).astype(int)
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train_scaled, y_train)
        elif model_type == "K-Means Clustering":
            model = KMeans(n_clusters=3, random_state=42, n_init=10)
            model.fit(X_train_scaled)
        elif model_type == "XGBoost":
            model = xgb.XGBRegressor(random_state=42, n_estimators=100)
            model.fit(X_train_scaled, y_train)
        
        predictions = model.predict(X_test_scaled)
        return {
            'model': model,
            'y_test': y_test,
            'y_pred': predictions,
            'X_test': X_test,
            'scaler': scaler
        }
    except Exception as e:
        st.error(f"🚫 Error training model: {str(e)}")
        return None

# Check dependencies
check_dependencies()

# Theme Configuration
THEMES = {
    "Zombie Theme": {
        "emojis": ["🧟", "🔪", "🩸"],
        "primary": "#8B0000",
        "background": "#1C2526",
        "card": "rgba(50, 50, 50, 0.7)",
        "text": "#D3D3D3",
        "button": "#8B0000",
        "button_hover": "#A52A2A",
        "font": "'Creepster', cursive",
        "gif": "https://media1.giphy.com/media/14a43iGArvJlPG/giphy.webp?cid=ecf05e47knhdz2omv1wz9e9nxbr4ajc2fz0qldqluosq45z4&ep=v1_gifs_related&rid=giphy.webp&ct=g",
        "ml_model": "Linear Regression",
        "feature": "Returns"
    },
    "Futuristic Theme": {
        "emojis": ["🚀", "🌌", "🤖"],
        "primary": "#00B7EB",
        "background": "#0A0B1E",
        "card": "rgba(20, 40, 80, 0.6)",
        "text": "#E6E6FA",
        "button": "#00B7EB",
        "button_hover": "#1E90FF",
        "font": "'Exo 2', sans-serif",
        "gif": "https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExdXV6ejlkNTA1ZGV6YXM2anlhcm55ZThiYzl0YnpzbjZkZDJ2cGFxdyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/xUOxfmkQrYVOr5qxP2/giphy.gif",
        "ml_model": "Logistic Regression",
        "feature": "RSI"
    },
    "Game of Thrones Theme": {
        "emojis": ["🏰", "⚔", "🛡"],
        "primary": "#FFD700",
        "background": "#2F2F2F",
        "card": "rgba(80, 60, 40, 0.7)",
        "text": "#F5F5DC",
        "button": "#FFD700",
        "button_hover": "#FFEC8B",
        "font": "'Cinzel', serif",
        "gif": "https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExMzY3ajRvejFwdnpmNmh1bXFnaDd1ZTNsYmJiajh3d3RidnhxNWt4ZiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7qDVAdWAtfLvFGGA/giphy.gif",
        "ml_model": "K-Means Clustering",
        "feature": "Moving Average"
    },
    "Nebula Pulse": {
        "emojis": ["🎯", "🔫", "🕹"],
        "primary": "#7B2CBF",
        "background": "#0D1B2A",
        "card": "rgba(30, 50, 80, 0.6)",
        "text": "#E0E1DD",
        "button": "#7B2CBF",
        "button_hover": "#9D4EDD",
        "font": "'Poppins', sans-serif",
        "gif": "https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExMjk5dG55ZnJ3cWN2bTc0MzF1M3BpemM3Z3cyNTQzcDN0M2Y3bWNuZiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/73WrT6n6yoVfq/giphy.gif",
        "ml_model": "XGBoost",
        "feature": "Returns"
    }
}

# Page Configuration
st.set_page_config(
    page_title="Nebula Finance Lab",
    layout="wide",
    page_icon="🪐",
    initial_sidebar_state="expanded"
)

# Initialize Session State
session_defaults = {
    'data': {},
    'current_theme': "Nebula Pulse",
    'model_trained': False,
    'predictions': None,
    'features': [],
    'target': None
}
for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Dynamic Styling
def apply_theme_styling(theme):
    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&family=Creepster&family=Exo+2:wght@300;400;600&family=Cinzel:wght@400&display=swap');
        :root {{
            --primary: {theme['primary']};
            --background: {theme['background']};
            --card: {theme['card']};
            --text: {theme['text']};
            --button: {theme['button']};
            --button-hover: {theme['button_hover']};
        }}
        .stApp {{
            background: var(--background);
            color: var(--text);
            animation: themeFade 1s ease-in-out;
        }}
        @keyframes themeFade {{
            0% {{ opacity: 0; }}
            100% {{ opacity: 1; }}
        }}
        [data-testid="stSidebar"] > div:first-child {{
            background: var(--background);
            padding: 20px;
            border-right: 2px solid var(--primary);
        }}
        .theme-card {{
            background: var(--card);
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }}
        .stButton>button {{
            background: var(--button);
            color: var(--text);
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-family: {theme['font']};
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            background: var(--button-hover);
            box-shadow: 0 0 10px var(--primary);
        }}
        h1, h2, h3 {{
            font-family: {theme['font']};
            color: var(--primary);
        }}
        p, div, input, label, .stMarkdown {{
            font-family: {theme['font']};
            color: var(--text);
        }}
        .stTextInput input, .stSelectbox > div > div {{
            background: rgba(50, 70, 100, 0.5);
            color: var(--text);
            border: 1px solid var(--primary);
            border-radius: 8px;
        }}
        .js-plotly-plot .plotly {{
            background: var(--card) !important;
            border-radius: 10px;
        }}
        .stAlert {{
            background: var(--card);
            border-left: 4px solid var(--primary);
            border-radius: 8px;
            color: var(--text);
        }}
        .theme-emoji {{
            font-size: 2rem;
            animation: float 2s ease-in-out infinite;
        }}
        @keyframes float {{
            0% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(-10px); }}
            100% {{ transform: translateY(0px); }}
        }}
    </style>
    """

# Sidebar
with st.sidebar:
    st.session_state.current_theme = st.selectbox(
        "🎨 Theme",
        list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.current_theme)
    )
    theme = THEMES[st.session_state.current_theme]
    st.markdown(apply_theme_styling(theme), unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="text-align: center;">
        <span class="theme-emoji">{theme['emojis'][0]}</span>
        <h2 style='color: var(--primary);'>Nebula Finance Lab</h2>
        <p>Explore stocks with {theme['ml_model']}</p>
        <img src="{theme['gif']}" width="100%" style="border-radius: 10px;">
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader(f"{theme['emojis'][1]} Load Data")
    data_source = st.radio("Source", ["Yahoo Finance", "Upload File"])
    
    if data_source == "Yahoo Finance":
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            ticker = st.text_input("Ticker (e.g., AAPL)", "AAPL")
        with col2:
            period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=3)
        with col3:
            refresh = st.button("🔄 Refresh")
            clear_cache = st.button("🗑 Clear Cache")
        
        if clear_cache:
            st.cache_data.clear()
            st.success("✅ Cache cleared!")
        
        if period in ["1mo", "3mo"]:
            st.info("💡 Tip: Short periods may return limited data. Using '1y' or 'max' can improve results.")
        
        if st.button("Fetch Data", key="fetch_data") or refresh:
            if not ticker or not ticker.strip():
                st.error("🚫 Please enter a ticker symbol")
            else:
                with st.spinner(f"Fetching data for {ticker}..."):
                    try:
                        st.session_state.error = None
                        df = download_stock_data(ticker, period)
                        if df is not None and not df.empty:
                            df = flatten_multiindex_columns(df)
                            df = process_dataframe(df, theme['feature'])
                            if df is not None:
                                valid, message = validate_data(df, theme['feature'])
                                if valid:
                                    st.session_state.data[ticker] = df
                                    st.session_state.model_trained = False
                                    st.success(f"✅ Loaded {ticker} data ({len(df)} rows)")
                                    st.info(f"📊 Data range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
                                else:
                                    st.error(f"⚠ {message}")
                            else:
                                st.error("🚫 No data after processing. Try a different period or ticker.")
                        else:
                            st.error(f"🚫 No data retrieved for {ticker}. See suggestions above.")
                    except Exception as e:
                        st.error(f"🚫 Error during data fetch: {str(e)}")
                        st.info("💡 Tip: Try using a longer period (e.g., '1y' or 'max') or a different ticker.")
    
    else:
        uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
        if uploaded_file:
            with st.spinner("Processing file..."):
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)
                    elif df.index.name == 'Date':
                        df.index = pd.to_datetime(df.index)
                    else:
                        st.error("🚫 File must contain a 'Date' column")
                        st.stop()
                    
                    df = flatten_multiindex_columns(df)
                    df = process_dataframe(df, theme['feature'])
                    if df is not None:
                        valid, message = validate_data(df, theme['feature'])
                        if valid:
                            st.session_state.data['Uploaded'] = df
                            st.session_state.model_trained = False
                            st.success(f"✅ Loaded file ({len(df)} rows)")
                        else:
                            st.error(f"⚠ {message}")
                    else:
                        st.error("🚫 No data after processing uploaded file.")
                except Exception as e:
                    st.error(f"🚫 Error processing file: {str(e)}")
                    st.error("Stack trace:")
                    st.code(traceback.format_exc())

# Main Content
st.markdown(f"<h1>{theme['emojis'][0]} Nebula Finance Lab</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color: var(--text);'>Welcome! Load data to analyze stocks with {theme['ml_model']} and {theme['feature']}.</p>", unsafe_allow_html=True)

if st.session_state.data:
    tabs = st.tabs([f"{theme['emojis'][1]} Dashboard", f"{theme['emojis'][2]} Analysis"])
    
    with tabs[0]:
        st.markdown("<div class='theme-card'>", unsafe_allow_html=True)
        ticker = st.selectbox("Select Data", list(st.session_state.data.keys()))
        df = st.session_state.data[ticker]
        
        st.subheader("Data Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", len(df.columns))
        col3.metric("Dates", f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
        
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            plot_col = st.selectbox("Plot Column", numeric_cols)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df[plot_col], mode='lines', line=dict(color=theme['primary'])))
            fig.update_layout(
                title=f"{plot_col} Trend",
                plot_bgcolor=theme['card'],
                paper_bgcolor=theme['background'],
                font_color=theme['text'],
                font=dict(family=theme['font'])
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Statistics")
            stats = df[numeric_cols].describe()
            st.dataframe(stats.style.format("{:.2f}"), height=300)
        else:
            st.warning("No numeric columns available.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown("<div class='theme-card'>", unsafe_allow_html=True)
        ticker = st.selectbox("Select Data", list(st.session_state.data.keys()), key="analysis_ticker")
        df = st.session_state.data[ticker].copy()
        
        st.subheader(f"Feature: {theme['feature']}")
        if 'Close' not in df.columns:
            st.error("⚠ Dataset must contain a 'Close' column.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()
        feature = theme['feature']
        try:
            if feature == "Returns":
                df['Returns'] = df['Close'].pct_change()
                df = df.dropna()
            elif feature == "RSI":
                if len(df) >= 14:
                    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
                    df = df.dropna()
                else:
                    st.error("⚠ Need at least 14 rows for RSI.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.stop()
            elif feature == "Moving Average":
                if len(df) >= 20:
                    df['MA20'] = df['Close'].rolling(window=20).mean()
                    df = df.dropna()
                else:
                    st.error("⚠ Need at least 20 rows for Moving Average.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.stop()
            
            if df.empty or len(df) < 2:
                st.error(f"⚠ No valid data after generating {feature}.")
                st.markdown("</div>", unsafe_allow_html=True)
                st.stop()
            
            st.session_state.data[ticker] = df
            fig = px.line(df, x=df.index, y=feature, title=f"{feature} Over Time")
            fig.update_layout(
                plot_bgcolor=theme['card'],
                paper_bgcolor=theme['background'],
                font_color=theme['text'],
                font=dict(family=theme['font'])
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader(f"Model: {theme['ml_model']}")
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            target = st.selectbox("Target", numeric_cols, index=len(numeric_cols)-1)
            features = [feature]
            
            if st.button("Train Model", key="train_model", use_container_width=True):
                with st.spinner("Training model..."):
                    try:
                        X = df[features]
                        y = df[target]
                        model_results = train_model(X, y, theme['ml_model'])
                        if model_results:
                            st.session_state.predictions = {
                                'y_test': model_results['y_test'],
                                'y_pred': model_results['y_pred'],
                                'X_test': model_results['X_test']
                            }
                            st.session_state.model_trained = True
                            st.success("✅ Model trained!")
                            st.balloons()
                    except Exception as e:
                        st.error(f"🚫 Error during model training: {str(e)}")
                        st.error("Stack trace:")
                        st.code(traceback.format_exc())
            
            if st.session_state.model_trained and st.session_state.predictions:
                st.subheader("Results")
                y_test = st.session_state.predictions['y_test']
                y_pred = st.session_state.predictions['y_pred']
                
                if theme['ml_model'] == "Linear Regression":
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = r2_score(y_test, y_pred)
                    col1, col2 = st.columns(2)
                    col1.metric("RMSE", f"{rmse:.2f}")
                    col2.metric("R² Score", f"{r2:.2f}")
                    
                    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).reset_index(drop=True)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=results.index, y=results['Actual'], name='Actual', mode='markers', marker=dict(color=theme['primary'])))
                    fig.add_trace(go.Scatter(x=results.index, y=results['Predicted'], name='Predicted', mode='markers', marker=dict(color=theme['button'])))
                    fig.update_layout(
                        title="Actual vs Predicted",
                        plot_bgcolor=theme['card'],
                        paper_bgcolor=theme['background'],
                        font_color=theme['text'],
                        font=dict(family=theme['font'])
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif theme['ml_model'] == "Logistic Regression":
                    accuracy = accuracy_score(y_test, y_pred)
                    st.metric("Accuracy", f"{accuracy:.2f}")
                    
                    fig = px.histogram(x=y_pred, title="Prediction Distribution (0: Negative, 1: Positive)")
                    fig.update_layout(
                        plot_bgcolor=theme['card'],
                        paper_bgcolor=theme['background'],
                        font_color=theme['text'],
                        font=dict(family=theme['font'])
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).reset_index(drop=True)
                
                elif theme['ml_model'] == "K-Means Clustering":
                    clusters = y_pred
                    cluster_sizes = np.bincount(clusters, minlength=3)
                    st.write("Cluster Sizes:", dict(zip(range(3), cluster_sizes)))
                    
                    results = pd.DataFrame({'Cluster': clusters}, index=st.session_state.predictions['X_test'].index).reset_index()
                    fig = px.histogram(results, x='Cluster', title="Cluster Distribution")
                    fig.update_layout(
                        plot_bgcolor=theme['card'],
                        paper_bgcolor=theme['background'],
                        font_color=theme['text'],
                        font=dict(family=theme['font'])
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                else:  # XGBoost
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = r2_score(y_test, y_pred)
                    col1, col2 = st.columns(2)
                    col1.metric("RMSE", f"{rmse:.2f}")
                    col2.metric("R² Score", f"{r2:.2f}")
                    
                    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).reset_index(drop=True)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=results.index, y=results['Actual'], name='Actual', mode='markers', marker=dict(color=theme['primary'])))
                    fig.add_trace(go.Scatter(x=results.index, y=results['Predicted'], name='Predicted', mode='markers', marker=dict(color=theme['button'])))
                    fig.update_layout(
                        title="Actual vs Predicted",
                        plot_bgcolor=theme['card'],
                        paper_bgcolor=theme['background'],
                        font_color=theme['text'],
                        font=dict(family=theme['font'])
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "💾 Download Results",
                    csv,
                    "predictions.csv",
                    "text/csv",
                    key="download_preds"
                )
        
        except Exception as e:
            st.error(f"🚫 Error: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(f"""
<div style="text-align: center; color: {theme['text']}; font-size: 12px; margin-top: 30px;">
    <p>Nebula Finance Lab © {datetime.now().year}</p>
</div>
""", unsafe_allow_html=True)
