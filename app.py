
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import os
from supabase import create_client, Client
from dotenv import load_dotenv
import pytz

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PSX Stock Analyzer Cloud",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with cloud theme
st.markdown("""
<style>
    /* Cloud-themed animations */
    @keyframes cloudFloat {
        0% { transform: translateY(0px) }
        50% { transform: translateY(-5px) }
        100% { transform: translateY(0px) }
    }
    
    @keyframes gradientFlow {
        0% { background-position: 0% 50% }
        50% { background-position: 100% 50% }
        100% { background-position: 0% 50% }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .cloud-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #1e90ff, #4169e1, #87ceeb);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: 
            gradientFlow 5s ease infinite,
            fadeInUp 1s ease-out;
    }
    
    .cloud-subheader {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        animation: fadeInUp 1.2s ease-out;
        animation-fill-mode: both;
    }
    
    .cloud-card {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out, cloudFloat 6s ease-in-out infinite;
    }
    
    .cloud-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        animation-play-state: paused;
    }
    
    .time-bar-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.8rem;
        margin: 0.5rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .time-bar-card:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .time-bar-card.active {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
    }
    
    .volume-delta-positive {
        color: #4CAF50;
        font-weight: bold;
        background: rgba(76, 175, 80, 0.1);
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
    }
    
    .volume-delta-negative {
        color: #f44336;
        font-weight: bold;
        background: rgba(244, 67, 54, 0.1);
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
    }
    
    .data-freshness-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    .data-freshness-indicator.fresh {
        background-color: #4CAF50;
    }
    
    .data-freshness-indicator.stale {
        background-color: #FF9800;
    }
    
    .data-freshness-indicator.old {
        background-color: #f44336;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.1); opacity: 0.7; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .interval-badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0 0.2rem;
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    """Initialize Supabase client"""
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            st.error("Supabase credentials not found. Please set SUPABASE_URL and SUPABASE_KEY in .env file")
            return None
        
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        st.error(f"Error initializing Supabase: {str(e)}")
        return None

# Initialize
supabase = init_supabase()

# Constants
TIMEZONE = pytz.timezone('Asia/Karachi')
INTERVALS = {
    '5m': 5,
    '15m': 15,
    '1h': 60,
    '4h': 240,
    '1d': 1440
}

class DataManager:
    """Manages data fetching and aggregation from Supabase"""
    
    @staticmethod
    def get_available_timestamps(interval='5m'):
        """Get available timestamps for the selected interval"""
        try:
            if supabase is None:
                return []
            
            # Get all unique scrape times
            response = supabase.table('stock_data')\
                .select('scrape_time')\
                .order('scrape_time', desc=True)\
                .execute()
            
            if not response.data:
                return []
            
            # Convert to datetime objects
            timestamps = [datetime.fromisoformat(item['scrape_time'].replace('Z', '+00:00')) 
                         for item in response.data]
            
            # Filter based on interval
            if interval == '5m':
                return timestamps
            elif interval == '15m':
                # Get every 15 minutes
                return [ts for ts in timestamps if ts.minute % 15 == 0]
            elif interval == '1h':
                # Get every hour
                return [ts for ts in timestamps if ts.minute == 0]
            elif interval == '4h':
                # Get every 4 hours
                return [ts for ts in timestamps if ts.hour % 4 == 0 and ts.minute == 0]
            elif interval == '1d':
                # Get daily (only one per day)
                daily_timestamps = []
                seen_days = set()
                for ts in timestamps:
                    day_key = ts.date()
                    if day_key not in seen_days:
                        daily_timestamps.append(ts)
                        seen_days.add(day_key)
                return daily_timestamps
            
            return timestamps
            
        except Exception as e:
            st.error(f"Error fetching timestamps: {str(e)}")
            return []
    
    @staticmethod
    def fetch_data_for_timestamp(timestamp, interval='5m'):
        """Fetch and aggregate data for a specific timestamp and interval"""
        try:
            if supabase is None:
                return None
            
            # Convert to UTC string for query
            timestamp_utc = timestamp.astimezone(pytz.UTC)
            start_time = timestamp_utc
            end_time = timestamp_utc
            
            # Adjust time window based on interval
            if interval == '15m':
                start_time = timestamp_utc - timedelta(minutes=15)
            elif interval == '1h':
                start_time = timestamp_utc - timedelta(minutes=60)
            elif interval == '4h':
                start_time = timestamp_utc - timedelta(minutes=240)
            elif interval == '1d':
                # Assuming trading day is 6 hours
                start_time = timestamp_utc - timedelta(minutes=360)
            
            # Query data within the time window
            response = supabase.table('stock_data')\
                .select('*')\
                .gte('scrape_time', start_time.isoformat())\
                .lte('scrape_time', end_time.isoformat())\
                .execute()
            
            if not response.data:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(response.data)
            
            # Aggregate if needed (for intervals > 5m)
            if interval != '5m':
                df = DataManager.aggregate_data(df, interval, timestamp_utc)
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None
    
    @staticmethod
    def aggregate_data(df, interval, timestamp):
        """Aggregate 5-minute data to higher timeframes"""
        if df.empty:
            return df
        
        aggregated = []
        
        # Group by symbol
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol]
            
            if symbol_data.empty:
                continue
            
            # Calculate aggregated values
            agg_row = {
                'symbol': symbol,
                'sector': symbol_data.iloc[0]['sector'] if 'sector' in symbol_data.columns else '',
                'listed_in': symbol_data.iloc[0]['listed_in'] if 'listed_in' in symbol_data.columns else '',
                'ldcp': symbol_data.iloc[0]['ldcp'] if len(symbol_data) > 0 else 0,
                'open': symbol_data.iloc[0]['open'] if len(symbol_data) > 0 else 0,
                'high': symbol_data['high'].max() if len(symbol_data) > 0 else 0,
                'low': symbol_data['low'].min() if len(symbol_data) > 0 else 0,
                'current': symbol_data.iloc[-1]['current'] if len(symbol_data) > 0 else 0,
                'volume': symbol_data['volume'].sum() if len(symbol_data) > 0 else 0,
                'scrape_time': timestamp.isoformat()
            }
            
            # Calculate change and change_percent
            if agg_row['open'] > 0:
                agg_row['change'] = agg_row['current'] - agg_row['open']
                agg_row['change_percent'] = (agg_row['change'] / agg_row['open']) * 100
            else:
                agg_row['change'] = 0
                agg_row['change_percent'] = 0
            
            aggregated.append(agg_row)
        
        return pd.DataFrame(aggregated)
    
    @staticmethod
    def calculate_volume_delta(current_df, interval, timestamp):
        """Calculate volume delta compared to previous interval"""
        if current_df is None or current_df.empty:
            return current_df
        
        # Get previous timestamp
        if interval == '5m':
            prev_timestamp = timestamp - timedelta(minutes=5)
        elif interval == '15m':
            prev_timestamp = timestamp - timedelta(minutes=15)
        elif interval == '1h':
            prev_timestamp = timestamp - timedelta(hours=1)
        elif interval == '4h':
            prev_timestamp = timestamp - timedelta(hours=4)
        elif interval == '1d':
            prev_timestamp = timestamp - timedelta(days=1)
        else:
            return current_df
        
        # Fetch previous interval data
        prev_df = DataManager.fetch_data_for_timestamp(prev_timestamp, interval)
        
        if prev_df is None or prev_df.empty:
            # If no previous data, set delta to 0
            current_df['volume_delta'] = 0
            current_df['volume_delta_percent'] = 0
            return current_df
        
        # Calculate deltas
        deltas = []
        delta_percents = []
        
        for _, row in current_df.iterrows():
            symbol = row['symbol']
            current_volume = row['volume']
            
            # Find previous volume for this symbol
            prev_row = prev_df[prev_df['symbol'] == symbol]
            if not prev_row.empty:
                prev_volume = prev_row.iloc[0]['volume']
                volume_delta = current_volume - prev_volume
                if prev_volume > 0:
                    volume_delta_percent = (volume_delta / prev_volume) * 100
                else:
                    volume_delta_percent = 0
            else:
                volume_delta = current_volume  # No previous data, use current volume as delta
                volume_delta_percent = 100
            
            deltas.append(volume_delta)
            delta_percents.append(volume_delta_percent)
        
        current_df['volume_delta'] = deltas
        current_df['volume_delta_percent'] = delta_percents
        
        return current_df

def display_time_bar_selector():
    """Display time interval selector"""
    st.markdown("### ⏰ Time Interval Selection")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("5m", use_container_width=True, 
                    type="primary" if st.session_state.get('interval', '5m') == '5m' else "secondary"):
            st.session_state.interval = '5m'
            st.rerun()
    
    with col2:
        if st.button("15m", use_container_width=True,
                    type="primary" if st.session_state.get('interval', '5m') == '15m' else "secondary"):
            st.session_state.interval = '15m'
            st.rerun()
    
    with col3:
        if st.button("1h", use_container_width=True,
                    type="primary" if st.session_state.get('interval', '5m') == '1h' else "secondary"):
            st.session_state.interval = '1h'
            st.rerun()
    
    with col4:
        if st.button("4h", use_container_width=True,
                    type="primary" if st.session_state.get('interval', '5m') == '4h' else "secondary"):
            st.session_state.interval = '4h'
            st.rerun()
    
    with col5:
        if st.button("1d", use_container_width=True,
                    type="primary" if st.session_state.get('interval', '5m') == '1d' else "secondary"):
            st.session_state.interval = '1d'
            st.rerun()

def display_timestamp_selector():
    """Display available timestamps for the selected interval"""
    interval = st.session_state.get('interval', '5m')
    timestamps = DataManager.get_available_timestamps(interval)
    
    if not timestamps:
        st.warning(f"No data available for {interval} interval")
        return None
    
    # Format timestamps for display
    formatted_timestamps = []
    for ts in timestamps:
        local_ts = ts.astimezone(TIMEZONE)
        if interval == '1d':
            display = local_ts.strftime("%Y-%m-%d")
        else:
            display = local_ts.strftime("%Y-%m-%d %H:%M")
        formatted_timestamps.append((display, ts))
    
    # Create selector
    selected_display = st.selectbox(
        f"Select {interval} timestamp",
        options=[t[0] for t in formatted_timestamps],
        index=0,
        key=f"timestamp_selector_{interval}"
    )
    
    # Find corresponding timestamp
    selected_ts = next(t[1] for t in formatted_timestamps if t[0] == selected_display)
    
    return selected_ts

def display_stock_metrics_with_delta(df):
    """Display key stock market metrics with volume deltas"""
    if df is None or df.empty:
        return
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown('<div class="cloud-card">', unsafe_allow_html=True)
        st.metric("Total Stocks", len(df))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="cloud-card">', unsafe_allow_html=True)
        gainers = len(df[df['change_percent'] > 0]) if 'change_percent' in df.columns else 0
        st.metric("Gaining Stocks", gainers)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="cloud-card">', unsafe_allow_html=True)
        losers = len(df[df['change_percent'] < 0]) if 'change_percent' in df.columns else 0
        st.metric("Losing Stocks", losers)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="cloud-card">', unsafe_allow_html=True)
        total_volume = f"{df['volume'].sum():,}" if 'volume' in df.columns else "N/A"
        st.metric("Total Volume", total_volume)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown('<div class="cloud-card">', unsafe_allow_html=True)
        if 'volume_delta' in df.columns:
            total_delta = df['volume_delta'].sum()
            delta_sign = "+" if total_delta >= 0 else ""
            st.metric("Volume Δ", f"{delta_sign}{total_delta:,.0f}")
        else:
            st.metric("Volume Δ", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)

def format_volume_delta(row):
    """Format volume delta for display"""
    if 'volume_delta' not in row:
        return ""
    
    delta = row['volume_delta']
    delta_percent = row.get('volume_delta_percent', 0)
    
    if delta > 0:
        return f'<span class="volume-delta-positive">↑ {delta:+,.0f} ({delta_percent:+.1f}%)</span>'
    elif delta < 0:
        return f'<span class="volume-delta-negative">↓ {delta:+,.0f} ({delta_percent:+.1f}%)</span>'
    else:
        return f'<span>0 (0%)</span>'

def main():
    # Cloud-themed header
    st.markdown("""
    <div class="cloud-header">
        ☁️ PSX Cloud Stock Analyzer
    </div>
    <div class="cloud-subheader">
        Real-time market data with Supabase backend | Auto-updated every 5 minutes
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'interval' not in st.session_state:
        st.session_state.interval = '5m'
    if 'selected_timestamp' not in st.session_state:
        st.session_state.selected_timestamp = None
    if 'data_fetched' not in st.session_state:
        st.session_state.data_fetched = False
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="cloud-card">', unsafe_allow_html=True)
        st.title("⚡ Cloud Control Panel")
        st.markdown("---")
        
        # Connection status
        if supabase:
            st.success("✅ Connected to Supabase")
        else:
            st.error("❌ Not connected to Supabase")
            st.info("Please set up Supabase credentials in .env file")
        
        # Data freshness indicator
        st.markdown("### 📊 Data Status")
        if st.session_state.data_fetched and st.session_state.selected_timestamp:
            now = datetime.now(pytz.UTC)
            data_age = (now - st.session_state.selected_timestamp).total_seconds() / 60
            
            if data_age < 10:
                freshness = "fresh"
                message = f"🟢 Data from {st.session_state.selected_timestamp.astimezone(TIMEZONE).strftime('%H:%M')}"
            elif data_age < 30:
                freshness = "stale"
                message = f"🟡 Data from {st.session_state.selected_timestamp.astimezone(TIMEZONE).strftime('%H:%M')}"
            else:
                freshness = "old"
                message = f"🔴 Data from {st.session_state.selected_timestamp.astimezone(TIMEZONE).strftime('%H:%M')}"
            
            st.markdown(f'<span class="data-freshness-indicator {freshness}"></span>{message}', unsafe_allow_html=True)
        
        # Fetch button
        st.markdown("---")
        st.subheader("🔄 Data Operations")
        
        if st.button("📥 Fetch Latest Data", use_container_width=True, type="primary"):
            with st.spinner("Fetching latest data from Supabase..."):
                # Get latest timestamp
                timestamps = DataManager.get_available_timestamps(st.session_state.interval)
                if timestamps:
                    st.session_state.selected_timestamp = timestamps[0]
                    st.session_state.data_fetched = True
                    st.rerun()
                else:
                    st.error("No data available in Supabase")
        
        # Manual refresh
        if st.button("🔄 Refresh Current View", use_container_width=True):
            st.rerun()
        
        # Data info
        st.markdown("---")
        st.subheader("ℹ️ About")
        st.markdown("""
        **Cloud Features:**
        - Auto-scraping every 5 minutes
        - Supabase database storage
        - Time-based aggregation
        - Volume delta calculations
        - Real-time updates
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content
    if not supabase:
        st.error("""
        ## ⚠️ Setup Required
        
        Please set up Supabase:
        1. Create a Supabase project
        2. Create a table with this schema:
        ```sql
        CREATE TABLE stock_data (
            id BIGSERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            sector TEXT,
            listed_in TEXT,
            ldcp FLOAT,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            current FLOAT,
            change FLOAT,
            change_percent FLOAT,
            volume BIGINT,
            scrape_time TIMESTAMP WITH TIME ZONE NOT NULL,
            dataset_name TEXT,
            UNIQUE(symbol, scrape_time)
        );
        ```
        3. Add indexes:
        ```sql
        CREATE INDEX idx_scrape_time ON stock_data(scrape_time);
        CREATE INDEX idx_symbol_scrape_time ON stock_data(symbol, scrape_time);
        ```
        4. Add credentials to `.env` file
        """)
        return
    
    # Time interval selection
    display_time_bar_selector()
    
    # Timestamp selection
    selected_ts = display_timestamp_selector()
    
    if selected_ts:
        st.session_state.selected_timestamp = selected_ts
        
        # Fetch and display data
        with st.spinner(f"Loading {st.session_state.interval} data for {selected_ts.astimezone(TIMEZONE).strftime('%H:%M')}..."):
            # Fetch data
            df = DataManager.fetch_data_for_timestamp(selected_ts, st.session_state.interval)
            
            if df is not None and not df.empty:
                # Calculate volume deltas
                df = DataManager.calculate_volume_delta(df, st.session_state.interval, selected_ts)
                
                # Store in session state
                st.session_state.current_data = df
                st.session_state.data_fetched = True
                
                # Display metrics
                display_stock_metrics_with_delta(df)
                
                # Data filters
                st.markdown("---")
                st.subheader("🔍 Filter & Analyze")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    filter_by = st.selectbox("Filter by", ["All", "Gainers", "Losers", "High Volume", "Low Volume"])
                
                with col2:
                    sort_by = st.selectbox("Sort by", [
                        "Symbol (A-Z)", "Symbol (Z-A)",
                        "Change % (High-Low)", "Change % (Low-High)",
                        "Volume (High-Low)", "Volume (Low-High)"
                    ])
                
                with col3:
                    if st.button("Apply Filters", use_container_width=True):
                        pass  # Filters applied in display
                
                # Apply filters
                filtered_df = df.copy()
                
                if filter_by == "Gainers":
                    filtered_df = filtered_df[filtered_df['change_percent'] > 0]
                elif filter_by == "Losers":
                    filtered_df = filtered_df[filtered_df['change_percent'] < 0]
                elif filter_by == "High Volume":
                    filtered_df = filtered_df[filtered_df['volume'] > filtered_df['volume'].median()]
                elif filter_by == "Low Volume":
                    filtered_df = filtered_df[filtered_df['volume'] <= filtered_df['volume'].median()]
                
                # Apply sorting
                if sort_by == "Symbol (A-Z)":
                    filtered_df = filtered_df.sort_values('symbol')
                elif sort_by == "Symbol (Z-A)":
                    filtered_df = filtered_df.sort_values('symbol', ascending=False)
                elif sort_by == "Change % (High-Low)":
                    filtered_df = filtered_df.sort_values('change_percent', ascending=False)
                elif sort_by == "Change % (Low-High)":
                    filtered_df = filtered_df.sort_values('change_percent')
                elif sort_by == "Volume (High-Low)":
                    filtered_df = filtered_df.sort_values('volume', ascending=False)
                elif sort_by == "Volume (Low-High)":
                    filtered_df = filtered_df.sort_values('volume')
                
                # Display data
                st.markdown(f"### 📋 Stock Data ({len(filtered_df)} stocks)")
                
                if not filtered_df.empty:
                    # Format the DataFrame for display
                    display_df = filtered_df.copy()
                    
                    # Format numeric columns
                    numeric_cols = ['ldcp', 'open', 'high', 'low', 'current', 'change', 'change_percent', 'volume']
                    for col in numeric_cols:
                        if col in display_df.columns:
                            if col == 'volume':
                                display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}")
                            else:
                                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
                    
                    # Add volume delta column
                    if 'volume_delta' in display_df.columns and 'volume_delta_percent' in display_df.columns:
                        display_df['Volume Δ'] = display_df.apply(format_volume_delta, axis=1)
                    
                    # Display table
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        height=600,
                        column_config={
                            "Volume Δ": st.column_config.Column(
                                width="medium",
                                help="Volume change vs previous interval"
                            )
                        }
                    )
                    
                    # Download button
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Filtered Data",
                        data=csv,
                        file_name=f"psx_{st.session_state.interval}_{selected_ts.strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Visualizations
                    st.markdown("---")
                    st.subheader("📈 Data Visualizations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Top 10 by volume
                        if 'volume' in filtered_df.columns:
                            top_volume = filtered_df.nlargest(10, 'volume')[['symbol', 'volume']]
                            fig1 = px.bar(
                                top_volume,
                                x='symbol',
                                y='volume',
                                title='📊 Top 10 Stocks by Volume',
                                color='volume',
                                color_continuous_scale='Viridis'
                            )
                            st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        # Performance scatter
                        if all(col in filtered_df.columns for col in ['change_percent', 'volume', 'symbol']):
                            fig2 = px.scatter(
                                filtered_df,
                                x='volume',
                                y='change_percent',
                                size='current',
                                color='change_percent',
                                hover_name='symbol',
                                title='📈 Performance: Change % vs Volume',
                                color_continuous_scale='RdYlGn'
                            )
                            fig2.update_layout(xaxis_type="log")
                            st.plotly_chart(fig2, use_container_width=True)
                    
                    # Volume delta visualization
                    if 'volume_delta' in filtered_df.columns:
                        st.markdown("### 📊 Volume Deltas")
                        
                        # Get top positive and negative deltas
                        top_positives = filtered_df.nlargest(5, 'volume_delta')[['symbol', 'volume_delta', 'volume_delta_percent']]
                        top_negatives = filtered_df.nsmallest(5, 'volume_delta')[['symbol', 'volume_delta', 'volume_delta_percent']]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 📈 Largest Volume Increases")
                            for _, row in top_positives.iterrows():
                                st.markdown(f"""
                                <div class="cloud-card">
                                    <strong>{row['symbol']}</strong><br>
                                    Δ Volume: <span class="volume-delta-positive">+{row['volume_delta']:,.0f}</span><br>
                                    Δ %: {row['volume_delta_percent']:.1f}%
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("#### 📉 Largest Volume Decreases")
                            for _, row in top_negatives.iterrows():
                                st.markdown(f"""
                                <div class="cloud-card">
                                    <strong>{row['symbol']}</strong><br>
                                    Δ Volume: <span class="volume-delta-negative">{row['volume_delta']:,.0f}</span><br>
                                    Δ %: {row['volume_delta_percent']:.1f}%
                                </div>
                                """, unsafe_allow_html=True)
                
                else:
                    st.warning("No stocks match the filter criteria")
            
            else:
                st.error("No data available for the selected timestamp")
    else:
        # Welcome screen
        st.markdown("""
        <div class="cloud-card">
            <h3>👋 Welcome to PSX Cloud Stock Analyzer!</h3>
            <p>This cloud-based application provides real-time PSX stock market analysis with:</p>
            
            <h4>☁️ Cloud Features:</h4>
            <ul>
                <li><strong>Auto-scraping</strong>: Data updates every 5 minutes</li>
                <li><strong>Supabase Storage</strong>: All data stored in cloud database</li>
                <li><strong>Time-based Analysis</strong>: View 5m, 15m, 1h, 4h, 1D intervals</li>
                <li><strong>Volume Deltas</strong>: See volume changes between intervals</li>
                <li><strong>Real-time Updates</strong>: Always current data</li>
            </ul>
            
            <h4>🚀 Getting Started:</h4>
            <ol>
                <li>Select a time interval (5m, 15m, 1h, 4h, 1d)</li>
                <li>Choose a timestamp from the dropdown</li>
                <li>Apply filters and analyze the data</li>
                <li>Download filtered results</li>
            </ol>
            
            <div class="alert alert-info">
                <strong>Note:</strong> The scraper runs automatically every 5 minutes in the cloud.
                No manual scraping needed!
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()