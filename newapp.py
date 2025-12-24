import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import time as tm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import os
from supabase import create_client, Client
from dotenv import load_dotenv
import pytz
import streamlit.components.v1 as components

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PSX Stock Analyzer Cloud",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with cloud theme (keeping your existing CSS)
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
            gradientFlow 5s ease 0s 1 forwards,
            fadeInUp 1s ease-out 0s 1 forwards;
        animation-fill-mode: both;
    }
    
    .cloud-subheader {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        animation: fadeInUp 1.2s ease-out 0s 1 forwards;
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
        animation: fadeInUp 0.6s ease-out 0s 1 forwards, cloudFloat 6s ease-in-out 0s 1 forwards;
        animation-fill-mode: both;
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
    
    /* Header and Navigation Styles */
    .header-container {
        background: linear-gradient(135deg, #d32f2f 0%, #388e3c 100%);
        padding: 1rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-radius: 0.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(211, 47, 47, 0.3);
    }
    
    .logo-section {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .logo-icon {
        font-size: 2rem;
    }
    
    .logo-text {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 0;
        background: linear-gradient(45deg, #ff6b6b, #4CAF50, #ff9800, #2196F3);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientFlow 5s ease infinite;
    }
    
    .nav-menu-btn {
        background: rgba(255, 255, 255, 0.2);
        border: 2px solid rgba(255, 255, 255, 0.4);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        cursor: pointer;
        font-size: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .nav-menu-btn:hover {
        background: rgba(255, 255, 255, 0.3);
        border-color: rgba(255, 255, 255, 0.6);
    }
    
    .nav-menu-items {
        background: linear-gradient(135deg, #2c3e50 0%, #1a1a2e 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 0.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        padding: 0.5rem 0;
        margin-top: 0.5rem;
        backdrop-filter: blur(10px);
    }
    
    .nav-menu-item {
        padding: 0.75rem 1.5rem;
        color: white;
        text-decoration: none;
        display: block;
        transition: all 0.2s ease;
        border-left: 3px solid transparent;
    }
    
    .nav-menu-item:hover {
        background-color: rgba(255, 255, 255, 0.1);
        border-left-color: #4CAF50;
        padding-left: 1.8rem;
    }
    
    .nav-divider {
        height: 1px;
        background-color: rgba(255, 255, 255, 0.1);
        margin: 0.5rem 0;
    }
    
    /* Footer Styles */
    .footer-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 0.8rem;
        margin-top: 3rem;
        text-align: center;
    }
    
    .footer-content {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 2rem;
        margin-bottom: 2rem;
    }
    
    .footer-section {
        text-align: left;
    }
    
    .footer-section h4 {
        color: #e0e0ff;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    .footer-link {
        color: rgba(255, 255, 255, 0.85);
        text-decoration: none;
        display: block;
        padding: 0.5rem 0;
        transition: all 0.2s ease;
    }
    
    .footer-link:hover {
        color: white;
        padding-left: 0.5rem;
    }
    
    .footer-divider {
        height: 1px;
        background-color: rgba(255, 255, 255, 0.2);
        margin: 1.5rem 0;
    }
    
    .footer-bottom {
        text-align: center;
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.7);
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
PKT_TZ = pytz.timezone('Asia/Karachi')
TRADING_START = time(9, 30)  # 9:30 AM
TRADING_END = time(15, 30)   # 3:30 PM

class DataManager:
    """Manages data fetching and aggregation from Supabase"""
    
    @staticmethod
    def get_latest_trading_data():
        """Get the latest data within trading hours (9:30 AM - 3:30 PM PKT)"""
        try:
            if supabase is None:
                return None
            
            # Get current time in PKT
            now_pkt = datetime.now(PKT_TZ)
            
            # Determine if we're in trading hours
            current_time = now_pkt.time()
            
            # If outside trading hours, use the last trading day's data
            if current_time < TRADING_START or current_time > TRADING_END:
                # Get data from yesterday's trading session
                target_date = now_pkt - timedelta(days=1)
            else:
                target_date = now_pkt
            
            # Create start and end times for the trading day
            trading_start = datetime.combine(target_date.date(), TRADING_START)
            trading_end = datetime.combine(target_date.date(), TRADING_END)
            
            # Make timezone aware
            trading_start = PKT_TZ.localize(trading_start)
            trading_end = PKT_TZ.localize(trading_end)
            
            # Convert to UTC for database query
            trading_start_utc = trading_start.astimezone(pytz.UTC)
            trading_end_utc = trading_end.astimezone(pytz.UTC)
            
            # Query for data within trading hours
            response = supabase.table('stock_data')\
                .select('*')\
                .gte('scraped_at', trading_start_utc.isoformat())\
                .lte('scraped_at', trading_end_utc.isoformat())\
                .order('scraped_at', desc=True)\
                .limit(1000)\
                .execute()
            
            if not response.data:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(response.data)
            
            # Convert scraped_at to PKT
            if 'scraped_at' in df.columns:
                df['scraped_at'] = pd.to_datetime(df['scraped_at']).dt.tz_convert(PKT_TZ)
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching trading data: {str(e)}")
            return None
    
    @staticmethod
    def get_data_by_timestamp(target_timestamp):
        """Get data for a specific timestamp with tolerance of 1-2 minutes"""
        try:
            if supabase is None:
                return None
            
            # Add tolerance of 2 minutes
            start_time = target_timestamp - timedelta(minutes=2)
            end_time = target_timestamp + timedelta(minutes=2)
            
            # Convert to UTC
            start_time_utc = start_time.astimezone(pytz.UTC)
            end_time_utc = end_time.astimezone(pytz.UTC)
            
            # Query data
            response = supabase.table('stock_data')\
                .select('*')\
                .gte('scraped_at', start_time_utc.isoformat())\
                .lte('scraped_at', end_time_utc.isoformat())\
                .order('scraped_at', desc=True)\
                .limit(500)\
                .execute()
            
            if not response.data:
                return None
            
            # Get the closest timestamp
            data = response.data
            closest_item = min(data, key=lambda x: abs(
                pd.Timestamp(x['scraped_at']).tz_convert(PKT_TZ) - target_timestamp
            ))
            
            # Get all data for this exact timestamp
            exact_time = pd.Timestamp(closest_item['scraped_at']).tz_convert(PKT_TZ)
            
            exact_response = supabase.table('stock_data')\
                .select('*')\
                .gte('scraped_at', (exact_time - timedelta(seconds=30)).isoformat())\
                .lte('scraped_at', (exact_time + timedelta(seconds=30)).isoformat())\
                .execute()
            
            if not exact_response.data:
                return None
            
            df = pd.DataFrame(exact_response.data)
            
            # Convert scraped_at to PKT
            if 'scraped_at' in df.columns:
                df['scraped_at'] = pd.to_datetime(df['scraped_at']).dt.tz_convert(PKT_TZ)
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching data by timestamp: {str(e)}")
            return None
    
    @staticmethod
    def get_available_batches():
        """Get all available data batches (timestamps) from today's trading"""
        try:
            if supabase is None:
                return []
            
            # Get today's date in PKT
            today_pkt = datetime.now(PKT_TZ)
            
            # Create trading day boundaries
            trading_start = datetime.combine(today_pkt.date(), TRADING_START)
            trading_end = datetime.combine(today_pkt.date(), TRADING_END)
            
            # Make timezone aware
            trading_start = PKT_TZ.localize(trading_start)
            trading_end = PKT_TZ.localize(trading_end)
            
            # Convert to UTC
            trading_start_utc = trading_start.astimezone(pytz.UTC)
            trading_end_utc = trading_end.astimezone(pytz.UTC)
            
            # Query distinct timestamps
            response = supabase.table('stock_data')\
                .select('scraped_at')\
                .gte('scraped_at', trading_start_utc.isoformat())\
                .lte('scraped_at', trading_end_utc.isoformat())\
                .order('scraped_at', desc=True)\
                .execute()
            
            if not response.data:
                # Try yesterday if no data today
                yesterday = today_pkt - timedelta(days=1)
                trading_start = datetime.combine(yesterday.date(), TRADING_START)
                trading_end = datetime.combine(yesterday.date(), TRADING_END)
                trading_start = PKT_TZ.localize(trading_start)
                trading_end = PKT_TZ.localize(trading_end)
                trading_start_utc = trading_start.astimezone(pytz.UTC)
                trading_end_utc = trading_end.astimezone(pytz.UTC)
                
                response = supabase.table('stock_data')\
                    .select('scraped_at')\
                    .gte('scraped_at', trading_start_utc.isoformat())\
                    .lte('scraped_at', trading_end_utc.isoformat())\
                    .order('scraped_at', desc=True)\
                    .execute()
            
            if not response.data:
                return []
            
            # Extract and deduplicate timestamps
            timestamps = []
            seen_times = set()
            
            for item in response.data:
                ts = pd.Timestamp(item['scraped_at']).tz_convert(PKT_TZ)
                # Round to nearest minute for grouping
                rounded_ts = ts.replace(second=0, microsecond=0)
                if rounded_ts not in seen_times:
                    seen_times.add(rounded_ts)
                    timestamps.append(ts)
            
            return sorted(timestamps, reverse=True)
            
        except Exception as e:
            st.error(f"Error fetching available batches: {str(e)}")
            return []
    
    @staticmethod
    def format_data_for_display(df):
        """Format the DataFrame to show only 11 relevant columns"""
        if df is None or df.empty:
            return df
        
        # Create a copy to avoid modifying original
        display_df = df.copy()
        
        # Define the 11 columns we want to show (matching your CSV)
        display_columns = {
            'symbol': 'Symbol',
            'sector': 'Sector',
            'listed_in': 'Listed_In',
            'ldcp': 'LDCP',
            'open_price': 'Open',
            'high': 'High',
            'low': 'Low',
            'current_price': 'Current',
            'change': 'Change',
            'change_percent': 'Change(%)',
            'volume': 'Volume'
        }
        
        # Rename columns to match CSV format
        display_df = display_df.rename(columns=display_columns)
        
        # Select only the columns we want to display
        column_order = [
            'Symbol', 'Sector', 'Listed_In', 'LDCP', 'Open', 'High', 
            'Low', 'Current', 'Change', 'Change(%)', 'Volume'
        ]
        
        # Only include columns that exist
        existing_columns = [col for col in column_order if col in display_df.columns]
        
        return display_df[existing_columns]
    
    @staticmethod
    def calculate_market_metrics(df):
        """Calculate market metrics from the data"""
        if df is None or df.empty:
            return {}
        
        metrics = {
            'total_stocks': len(df),
            'gainers': len(df[df['change_percent'] > 0]) if 'change_percent' in df.columns else 0,
            'losers': len(df[df['change_percent'] < 0]) if 'change_percent' in df.columns else 0,
            'unchanged': len(df[df['change_percent'] == 0]) if 'change_percent' in df.columns else 0,
            'total_volume': df['volume'].sum() if 'volume' in df.columns else 0,
            'avg_change': df['change_percent'].mean() if 'change_percent' in df.columns else 0,
            'top_gainer': df.loc[df['change_percent'].idxmax()] if 'change_percent' in df.columns and not df.empty else None,
            'top_loser': df.loc[df['change_percent'].idxmin()] if 'change_percent' in df.columns and not df.empty else None,
            'most_active': df.loc[df['volume'].idxmax()] if 'volume' in df.columns and not df.empty else None
        }
        
        return metrics

def display_header_with_nav():
    """Display professional header with navigation menu"""
    # Initialize menu state
    if 'menu_open' not in st.session_state:
        st.session_state.menu_open = False
    
    col1, col2 = st.columns([0.9, 0.1])
    
    with col1:
        st.markdown("""
        <div class="header-container">
            <div class="logo-section">
                <span class="logo-icon">☁️</span>
                <h1 class="logo-text">PSX Cloud Stock Analyzer</h1>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Menu button using Unicode characters
        if st.button("⋮", key="menu_btn", help="More options"):
            st.session_state.menu_open = not st.session_state.menu_open
    
    # Display menu when open
    if st.session_state.menu_open:
        menu_html = """
        <div class="nav-menu-items">
            <a href="?" class="nav-menu-item">🏠 Home</a>
            <div class="nav-divider"></div>
            <a href="https://www.kaggle.com/wasiqaliyasir" target="_blank" class="nav-menu-item">ℹ️ About Us</a>
            <div class="nav-divider"></div>
            <a href="https://www.psx.com" target="_blank" class="nav-menu-item">📞 Support</a>
            <div class="nav-divider"></div>
            <a href="mailto:wasiqtaha@gmail.com" class="nav-menu-item">✉️ Contact Us</a>
        </div>
        """
        st.markdown(menu_html, unsafe_allow_html=True)

def display_footer():
    """Display professional footer with navigation and information"""
    footer_html = """
    <div class="footer-container">
        <div class="footer-content">
            <div class="footer-section">
                <h4>Home Page</h4>
                <a href="?" class="footer-link">Home</a>
                <a href="https://www.kaggle.com/wasiqaliyasir" target="_blank" class="footer-link">About Us</a>
                <a href="https://www.psx.com" target="_blank" class="footer-link">Support</a>
                <a href="mailto:wasiqtaha@gmail.com" class="footer-link">Contact Us</a>
            </div>
            <div class="footer-section">
                <h4>📊 Features</h4>
                <p style="margin: 0; font-size: 0.95rem; color: rgba(255, 255, 255, 0.85);">View the data according to the time</p>
                <p style="margin: 0; font-size: 0.95rem; color: rgba(255, 255, 255, 0.85);">Filter the stock market data</p>
                <p style="margin: 0; font-size: 0.95rem; color: rgba(255, 255, 255, 0.85);">Real-time market analysis</p>
                 <p style="margin: 0; font-size: 0.95rem; color: rgba(255, 255, 255, 0.85);">Comming soon features ...</p>
                 <p style="margin: 0; font-size: 0.95rem; color: rgba(255, 255, 255, 0.85);">ALL candleistick chart in one click.</p>
                  <p style="margin: 0; font-size: 0.95rem; color: rgba(255, 255, 255, 0.85);">With MACD, RSI indicators</p>
                   <p style="margin: 0; font-size: 0.95rem; color: rgba(255, 255, 255, 0.85);">Ai Features, Prediction results</p>
            </div>
            <div class="footer-section">
                <h4>☁️ About</h4>
                <p style="margin: 0; font-size: 0.95rem; color: rgba(255, 255, 255, 0.85);">PSX Cloud Stock Analyzer is a professional-grade market analysis tool. Created by PSX insights.</p>
            </div>
        </div>
        <div class="footer-divider"></div>
        <div class="footer-bottom">
            <p style="margin: 0;">© 2025 PSX Cloud Stock Analyzer. All rights reserved.</p>
            <p style="margin: 0.5rem 0 0 0;">Powered by Streamlit</p>
        </div>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

def display_market_metrics(metrics):
    """Display market metrics cards"""
    if not metrics:
        return
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown('<div class="cloud-card">', unsafe_allow_html=True)
        st.metric("Total Stocks", f"{metrics['total_stocks']:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="cloud-card">', unsafe_allow_html=True)
        st.metric("Gaining", f"{metrics['gainers']:,}", delta=f"+{metrics['gainers']}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="cloud-card">', unsafe_allow_html=True)
        st.metric("Declining", f"{metrics['losers']:,}", delta=f"-{metrics['losers']}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="cloud-card">', unsafe_allow_html=True)
        st.metric("Unchanged", f"{metrics['unchanged']:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown('<div class="cloud-card">', unsafe_allow_html=True)
        volume_formatted = f"{metrics['total_volume']:,.0f}" if metrics['total_volume'] >= 1000 else f"{metrics['total_volume']:,.0f}"
        st.metric("Total Volume", volume_formatted)
        st.markdown('</div>', unsafe_allow_html=True)

def display_top_performers(metrics, df):
    """Display top gainers, losers, and most active stocks"""
    if df is None or df.empty:
        return
    
    st.markdown("### 🏆 Top Performers")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if metrics['top_gainer'] is not None:
            st.markdown('<div class="cloud-card">', unsafe_allow_html=True)
            st.markdown("#### 📈 Top Gainer")
            st.markdown(f"**{metrics['top_gainer']['symbol']}**")
            st.markdown(f"Price: {metrics['top_gainer']['current_price']:.2f}")
            st.markdown(f"Change: +{metrics['top_gainer']['change_percent']:.2f}%")
            st.markdown(f"Volume: {metrics['top_gainer']['volume']:,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if metrics['top_loser'] is not None:
            st.markdown('<div class="cloud-card">', unsafe_allow_html=True)
            st.markdown("#### 📉 Top Loser")
            st.markdown(f"**{metrics['top_loser']['symbol']}**")
            st.markdown(f"Price: {metrics['top_loser']['current_price']:.2f}")
            st.markdown(f"Change: {metrics['top_loser']['change_percent']:.2f}%")
            st.markdown(f"Volume: {metrics['top_loser']['volume']:,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        if metrics['most_active'] is not None:
            st.markdown('<div class="cloud-card">', unsafe_allow_html=True)
            st.markdown("#### 🔥 Most Active")
            st.markdown(f"**{metrics['most_active']['symbol']}**")
            st.markdown(f"Price: {metrics['most_active']['current_price']:.2f}")
            st.markdown(f"Change: {metrics['most_active']['change_percent']:.2f}%")
            st.markdown(f"Volume: {metrics['most_active']['volume']:,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Display professional header with navigation
    display_header_with_nav()
    
    # Cloud-themed subheader
    st.markdown("""
    <div class="cloud-subheader">
        Real-time PSX market data | Auto-updated every 5 minutes | Trading Hours: 9:30 AM - 3:30 PM PKT
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'selected_batch' not in st.session_state:
        st.session_state.selected_batch = None
    if 'available_batches' not in st.session_state:
        st.session_state.available_batches = []
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = None
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="cloud-card">', unsafe_allow_html=True)
        st.title("⚡ Cloud Control Panel")
        st.markdown("---")
        
        # Connection status
        if supabase:
            st.success("✅ Connected to PSX Cloud Database")
        else:
            st.error("❌ Database Connection Failed")
            st.info("Check your .env file for SUPABASE_URL and SUPABASE_KEY")
        
        # Refresh data button
        st.markdown("---")
        if st.button("🔄 Refresh Market Data", use_container_width=True, type="primary"):
            with st.spinner("Fetching latest market data..."):
                st.session_state.available_batches = DataManager.get_available_batches()
                if st.session_state.available_batches:
                    # Get the latest batch
                    latest_batch = st.session_state.available_batches[0]
                    st.session_state.selected_batch = latest_batch
                    st.session_state.current_data = DataManager.get_data_by_timestamp(latest_batch)
                    st.session_state.last_refresh = datetime.now(PKT_TZ)
                    st.rerun()
                else:
                    st.warning("No data available in database")
        
        # Batch selection
        st.markdown("---")
        st.subheader("📅 Select Data Batch")
        
        # Get available batches if not already loaded
        if not st.session_state.available_batches:
            st.session_state.available_batches = DataManager.get_available_batches()
        
        if st.session_state.available_batches:
            # Format batch times for display
            batch_options = []
            for batch in st.session_state.available_batches:
                formatted_time = batch.strftime("%Y-%m-%d %H:%M:%S")
                batch_options.append((formatted_time, batch))
            
            # Create selectbox
            selected_option = st.selectbox(
                "Available Batches",
                options=[opt[0] for opt in batch_options],
                index=0
            )
            
            # Find selected batch
            selected_batch = next(opt[1] for opt in batch_options if opt[0] == selected_option)
            
            # Load button for selected batch
            if st.button("📥 Load Selected Batch", use_container_width=True):
                with st.spinner(f"Loading data for {selected_option}..."):
                    st.session_state.selected_batch = selected_batch
                    st.session_state.current_data = DataManager.get_data_by_timestamp(selected_batch)
                    st.session_state.last_refresh = datetime.now(PKT_TZ)
                    st.rerun()
        else:
            st.info("No data batches available")
        
        # Last refresh info
        if st.session_state.last_refresh:
            st.markdown("---")
            st.markdown(f"**Last Refresh:**")
            st.markdown(f"{st.session_state.last_refresh.strftime('%H:%M:%S')}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area
    if not supabase:
        st.error("""
        ## ⚠️ Database Setup Required
        
        Please configure your Supabase connection:
        
        1. Create a `.env` file in your project directory
        2. Add your Supabase credentials:
        ```
        SUPABASE_URL=your_supabase_url
        SUPABASE_KEY=your_supabase_anon_key
        ```
        3. Restart the application
        
        **Note:** The scraper should be running to populate the database with data every 5 minutes.
        """)
        return
    
    # Display current data
    if st.session_state.current_data is not None and not st.session_state.current_data.empty:
        df = st.session_state.current_data
        
        # Display batch info
        if st.session_state.selected_batch:
            batch_time = st.session_state.selected_batch.strftime("%Y-%m-%d %H:%M:%S")
            st.success(f"📊 Displaying market data for: **{batch_time} PKT**")
        
        # Calculate metrics
        metrics = DataManager.calculate_market_metrics(df)
        
        # Display metrics
        display_market_metrics(metrics)
        
        # Display top performers
        display_top_performers(metrics, df)
        
        # Format data for display (11 columns)
        display_df = DataManager.format_data_for_display(df)
        
        # Data filters
        st.markdown("---")
        st.subheader("🔍 Filter & Analyze")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Sector filter
            if 'Sector' in display_df.columns:
                sectors = ['All'] + sorted(display_df['Sector'].dropna().unique().tolist())
                selected_sector = st.selectbox("Filter by Sector", sectors)
            
            # Change filter
            change_filter = st.selectbox(
                "Filter by Performance",
                ["All", "Gainers (+)", "Losers (-)", "Unchanged"]
            )
        
        with col2:
            # Search symbol
            search_symbol = st.text_input("🔎 Search Symbol", placeholder="Enter stock symbol...")
            
            # Sort options
            sort_by = st.selectbox(
                "Sort by",
                ["Symbol (A-Z)", "Symbol (Z-A)", 
                 "Change % (High to Low)", "Change % (Low to High)",
                 "Volume (High to Low)", "Volume (Low to High)",
                 "Current Price (High to Low)", "Current Price (Low to High)"]
            )
        
        # Apply filters
        filtered_df = display_df.copy()
        
        # Apply sector filter
        if 'Sector' in filtered_df.columns and selected_sector != 'All':
            filtered_df = filtered_df[filtered_df['Sector'] == selected_sector]
        
        # Apply change filter
        if 'Change(%)' in filtered_df.columns:
            if change_filter == "Gainers (+)":
                filtered_df = filtered_df[filtered_df['Change(%)'] > 0]
            elif change_filter == "Losers (-)":
                filtered_df = filtered_df[filtered_df['Change(%)'] < 0]
            elif change_filter == "Unchanged":
                filtered_df = filtered_df[filtered_df['Change(%)'] == 0]
        
        # Apply search filter
        if search_symbol:
            filtered_df = filtered_df[filtered_df['Symbol'].str.contains(search_symbol, case=False, na=False)]
        
        # Apply sorting
        if sort_by == "Symbol (A-Z)":
            filtered_df = filtered_df.sort_values('Symbol')
        elif sort_by == "Symbol (Z-A)":
            filtered_df = filtered_df.sort_values('Symbol', ascending=False)
        elif sort_by == "Change % (High to Low)":
            filtered_df = filtered_df.sort_values('Change(%)', ascending=False)
        elif sort_by == "Change % (Low to High)":
            filtered_df = filtered_df.sort_values('Change(%)')
        elif sort_by == "Volume (High to Low)":
            filtered_df = filtered_df.sort_values('Volume', ascending=False)
        elif sort_by == "Volume (Low to High)":
            filtered_df = filtered_df.sort_values('Volume')
        elif sort_by == "Current Price (High to Low)":
            filtered_df = filtered_df.sort_values('Current', ascending=False)
        elif sort_by == "Current Price (Low to High)":
            filtered_df = filtered_df.sort_values('Current')
        
        # Display data table
        st.markdown(f"### 📋 Market Data ({len(filtered_df)} stocks)")
        
        if not filtered_df.empty:
            # Format numeric columns for better display
            display_formatted = filtered_df.copy()
            
            # Format numeric columns
            if 'LDCP' in display_formatted.columns:
                display_formatted['LDCP'] = display_formatted['LDCP'].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
            if 'Open' in display_formatted.columns:
                display_formatted['Open'] = display_formatted['Open'].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
            if 'High' in display_formatted.columns:
                display_formatted['High'] = display_formatted['High'].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
            if 'Low' in display_formatted.columns:
                display_formatted['Low'] = display_formatted['Low'].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
            if 'Current' in display_formatted.columns:
                display_formatted['Current'] = display_formatted['Current'].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
            if 'Change' in display_formatted.columns:
                display_formatted['Change'] = display_formatted['Change'].apply(lambda x: f"{x:+,.2f}" if pd.notnull(x) else "N/A")
            if 'Change(%)' in display_formatted.columns:
                display_formatted['Change(%)'] = display_formatted['Change(%)'].apply(lambda x: f"{x:+,.2f}%" if pd.notnull(x) else "N/A")
            if 'Volume' in display_formatted.columns:
                display_formatted['Volume'] = display_formatted['Volume'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A")
            
            # Display the table
            st.dataframe(
                display_formatted,
                use_container_width=True,
                height=600,
                column_config={
                    "Symbol": st.column_config.Column(width="small"),
                    "Sector": st.column_config.Column(width="medium"),
                    "Listed_In": st.column_config.Column(width="medium"),
                    "LDCP": st.column_config.Column(width="small"),
                    "Open": st.column_config.Column(width="small"),
                    "High": st.column_config.Column(width="small"),
                    "Low": st.column_config.Column(width="small"),
                    "Current": st.column_config.Column(width="small"),
                    "Change": st.column_config.Column(width="small"),
                    "Change(%)": st.column_config.Column(width="small"),
                    "Volume": st.column_config.Column(width="medium")
                }
            )
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data",
                data=csv,
                file_name=f"psx_data_{datetime.now(PKT_TZ).strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Visualizations
            st.markdown("---")
            st.subheader("📈 Market Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Volume distribution
                if 'Volume' in filtered_df.columns and 'Symbol' in filtered_df.columns:
                    top_volume = filtered_df.nlargest(10, 'Volume')[['Symbol', 'Volume']].copy()
                    top_volume['Volume'] = pd.to_numeric(top_volume['Volume'], errors='coerce')
                    
                    fig1 = px.bar(
                        top_volume,
                        x='Symbol',
                        y='Volume',
                        title='📊 Top 10 Stocks by Volume',
                        color='Volume',
                        color_continuous_scale='Viridis'
                    )
                    fig1.update_layout(xaxis_title="Symbol", yaxis_title="Volume")
                    st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Performance scatter
                if all(col in filtered_df.columns for col in ['Change(%)', 'Volume', 'Symbol']):
                    scatter_df = filtered_df.copy()
                    scatter_df['Change(%)'] = pd.to_numeric(scatter_df['Change(%)'], errors='coerce')
                    scatter_df['Volume'] = pd.to_numeric(scatter_df['Volume'], errors='coerce')
                    scatter_df['Current'] = pd.to_numeric(scatter_df['Current'], errors='coerce')
                    
                    fig2 = px.scatter(
                        scatter_df,
                        x='Volume',
                        y='Change(%)',
                        size='Current',
                        color='Change(%)',
                        hover_name='Symbol',
                        title='📈 Performance: Change % vs Volume',
                        color_continuous_scale='RdYlGn',
                        hover_data=['Symbol', 'Change(%)', 'Volume', 'Current']
                    )
                    fig2.update_layout(
                        xaxis_title="Volume (log scale)",
                        yaxis_title="Change %",
                        xaxis_type="log"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            
            # Sector analysis
            st.markdown("### 🏢 Sector Analysis")
            if 'Sector' in filtered_df.columns and 'Change(%)' in filtered_df.columns:
                sector_df = filtered_df.copy()
                sector_df['Change(%)'] = pd.to_numeric(sector_df['Change(%)'], errors='coerce')
                
                sector_stats = sector_df.groupby('Sector').agg({
                    'Symbol': 'count',
                    'Change(%)': 'mean',
                    'Volume': 'sum'
                }).reset_index()
                sector_stats = sector_stats.rename(columns={'Symbol': 'Count'})
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig3 = px.bar(
                        sector_stats,
                        x='Sector',
                        y='Count',
                        title='📊 Stocks per Sector',
                        color='Count',
                        color_continuous_scale='Blues'
                    )
                    fig3.update_layout(xaxis_title="Sector", yaxis_title="Number of Stocks")
                    st.plotly_chart(fig3, use_container_width=True)
                
                with col2:
                    fig4 = px.bar(
                        sector_stats,
                        x='Sector',
                        y='Change(%)',
                        title='📈 Average Change % per Sector',
                        color='Change(%)',
                        color_continuous_scale='RdYlGn'
                    )
                    fig4.update_layout(xaxis_title="Sector", yaxis_title="Average Change %")
                    st.plotly_chart(fig4, use_container_width=True)
        
        else:
            st.warning("No stocks match the filter criteria")
    
    else:
        # Welcome/No data screen
        if st.session_state.available_batches:
            # Auto-load the latest batch
            latest_batch = st.session_state.available_batches[0]
            st.session_state.selected_batch = latest_batch
            st.session_state.current_data = DataManager.get_data_by_timestamp(latest_batch)
            st.session_state.last_refresh = datetime.now(PKT_TZ)
            st.rerun()
        else:
            # Show welcome message
            html = """
            <style>
            /* Minimal subset of app styles required for the welcome card + animations */
            @keyframes cloudFloat {0%{transform:translateY(0)}50%{transform:translateY(-6px)}100%{transform:translateY(0)}}
            @keyframes gradientFlow {0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
            @keyframes fadeInUp {from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
            .cloud-card{background:#fff;padding:20px;border-radius:12px;box-shadow:0 8px 30px rgba(0,0,0,0.08);max-width:1000px;margin:10px auto;font-family:inherit;
              animation: fadeInUp .6s ease-out 0s 1 forwards, cloudFloat 6s ease-in-out 0s 1 forwards;
              animation-fill-mode: both;
            }
            .cloud-header{font-size:2.25rem;font-weight:700;text-align:center;margin-bottom:8px;background:linear-gradient(45deg,#ff6b6b,#4CAF50,#ff9800,#2196F3);background-size:300% 300%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;
              animation: gradientFlow 5s ease infinite, fadeInUp 1s ease-out 0s 1 forwards;
              animation-fill-mode: both;
            }
            .cloud-subheader{font-size:1.1rem;color:#555;text-align:center;margin-bottom:12px;opacity:.95; animation:fadeInUp 1.1s ease-out 0s 1 forwards; animation-fill-mode: both;}
            .cloud-card h4{margin-top:14px}
            .cloud-card ul{margin-left:1.1rem}
            .cloud-card ol{margin-left:1.1rem}
            .alert{margin-top:14px;padding:10px;border-radius:8px;background:linear-gradient(90deg,#e8f4ff,#f0f9ff);border:1px solid #d6ecff;color:#083b66}
            @media (max-width:600px){.cloud-header{font-size:1.6rem}}
            </style>
            <div class="cloud-card">
              <div class="cloud-header">👋 Welcome to PSX Cloud Stock Analyzer!</div>
              <div class="cloud-subheader">This cloud-based application provides real-time PSX stock market analysis with:</div>
              <div>
                <h4>☁️ Cloud Features:</h4>
                <ul>
                  <li><strong>Auto-scraping</strong>: Data updates every 5 minutes</li>
                  <li><strong>Supabase Storage</strong>: All data stored in cloud database</li>
                  <li><strong>Trading Hours Data</strong>: 9:30 AM - 3:30 PM PKT</li>
                  <li><strong>Real-time Analysis</strong>: Always current market data</li>
                  <li><strong>Advanced Filtering</strong>: Filter by sector, performance, volume</li>
                </ul>
                <h4>🚀 Getting Started:</h4>
                <ol>
                  <li>Click "Refresh Market Data" in the sidebar</li>
                  <li>Select a data batch from available timestamps</li>
                  <li>Apply filters to analyze specific stocks or sectors</li>
                  <li>Download filtered results for offline analysis</li>
                </ol>
                <div class="alert"><strong>Note:</strong> The scraper runs automatically every 5 minutes during trading hours. Data is stored in Supabase cloud database.</div>
              </div>
            </div>
            """
            components.html(html, height=420, scrolling=True)
    
    # Display footer at the bottom
    st.markdown("---")
    display_footer()

if __name__ == "__main__":
    main()