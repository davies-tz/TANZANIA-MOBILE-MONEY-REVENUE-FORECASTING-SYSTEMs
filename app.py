"""
Tanzania Mobile Money Revenue Forecasting System
Streamlit Deployment Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="TZ Mobile Money Revenue Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS for styling
# ----------------------------
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .prediction-card {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin: 1rem 0;
        animation: fadeIn 1s;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .prediction-amount {
        font-size: 4rem;
        font-weight: bold;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-weight: bold;
        border-radius: 10px;
        font-size: 1.2rem;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        color: #666;
        padding: 2rem;
        background: #f8f9fa;
        border-radius: 10px;
        margin-top: 2rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
st.markdown("""
<div class="main-header">
    <h1>💰 Tanzania Mobile Money Revenue Forecasting System</h1>
    <p style="font-size: 1.2rem; margin-top: 1rem;">Powered by Machine Learning | Predict daily revenue for Tigo Pesa, M-Pesa & Airtel Money</p>
    <p style="font-size: 1rem; opacity: 0.9;">📍 Tanzania Context | 📊 Linear Regression vs Decision Tree | 🚀 Real-time Predictions</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# Load models with caching
# ----------------------------
@st.cache_resource
def load_models():
    """Load all necessary models and encoders"""
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        le_weekend = joblib.load('le_weekend.pkl')
        le_season = joblib.load('le_season.pkl')
        
        model_type = "Decision Tree" if hasattr(model, 'tree_') else "Linear Regression"
        
        return model, scaler, le_weekend, le_season, model_type
    except FileNotFoundError as e:
        st.error(f"⚠️ Model file not found: {e}")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"⚠️ Error loading models: {str(e)}")
        return None, None, None, None, None

model, scaler, le_weekend, le_season, model_type = load_models()

# ----------------------------
# Main app logic
# ----------------------------
if model is not None:
    # Sidebar for user input
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/money-transfer.png", width=100)
        st.title("📊 Input Parameters")
        st.markdown("---")
        
        # Day selection
        st.subheader("📅 Select Day")
        day_options = {
            "Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4,
            "Friday": 5, "Saturday": 6, "Sunday": 7
        }
        selected_day_name = st.selectbox("Day of Week", list(day_options.keys()), index=4)
        day = day_options[selected_day_name]
        is_weekend = "Yes" if day in [6, 7] else "No"
        
        # Month selection
        st.subheader("📆 Select Month")
        month_names = {
            "January": 1, "February": 2, "March": 3, "April": 4,
            "May": 5, "June": 6, "July": 7, "August": 8,
            "September": 9, "October": 10, "November": 11, "December": 12
        }
        selected_month_name = st.selectbox("Month", list(month_names.keys()), index=5)
        month = month_names[selected_month_name]
        season = "High" if month in [6, 7, 8, 12, 1] else "Low"
        
        # Previous revenue input
        st.subheader("💰 Previous Day Revenue")
        prev_revenue = st.number_input("Amount (TZS)", min_value=10_000_000,
                                       max_value=100_000_000,
                                       value=50_000_000,
                                       step=1_000_000, format="%d")
        
        # Predict button
        predict_button = st.button("🚀 Predict Revenue", use_container_width=True)

    # ----------------------------
    # Main content: Trends & Insights
    # ----------------------------
    col1, col2 = st.columns([2,1])

    with col1:
        st.subheader("📈 Revenue Trends Visualization")
        dates_2025 = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')
        viz_df = pd.DataFrame({'date': dates_2025})
        viz_df['day_of_week'] = viz_df['date'].dt.dayofweek + 1
        viz_df['month'] = viz_df['date'].dt.month
        viz_df['is_weekend'] = viz_df['day_of_week'].apply(lambda x: 'Yes' if x >= 6 else 'No')
        viz_df['season'] = viz_df['month'].apply(lambda x: 'High' if x in [6,7,8,12,1] else 'Low')
        viz_df['prev_day_revenue'] = 50_000_000

        weekend_encoded = viz_df['is_weekend'].map({'No': 0, 'Yes': 1})
        season_encoded = viz_df['season'].map({'Low': 0, 'High': 1})

        features = np.column_stack([
            viz_df['day_of_week'], viz_df['month'],
            weekend_encoded, season_encoded, viz_df['prev_day_revenue']
        ])
        features_scaled = scaler.transform(features)
        predictions = model.predict(features_scaled)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=viz_df['date'],
            y=predictions/1e6,
            mode='lines',
            name='Daily Revenue',
            line=dict(color='#667eea', width=3),
            hovertemplate='<b>%{x|%b %d, %Y}</b><br>Revenue: TZS %{y:.1f}M<extra></extra>'
        ))

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Model Performance")
        st.markdown(f"**Best Model:** {model_type}\n**R²:** 0.89 | **MAE:** ±2.1M TZS")

    # ----------------------------
    # Prediction logic
    # ----------------------------
    if predict_button:
        try:
            weekend_enc = le_weekend.transform([is_weekend])[0]
            season_enc = le_season.transform([season])[0]
            features = np.array([[day, month, weekend_enc, season_enc, prev_revenue]])
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            change = ((prediction - prev_revenue)/prev_revenue)*100

            st.markdown(f"""
            <div class="prediction-card">
                <h2>📊 Predicted Revenue</h2>
                <div class="prediction-amount">TZS {prediction:,.0f}</div>
                <p>{selected_day_name}, {selected_month_name}</p>
                <p>Change vs Previous Day: {change:+.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
else:
    st.error("⚠️ Failed to load model. Ensure all files are in the same directory:")
    st.write("- best_model.pkl\n- scaler.pkl\n- le_weekend.pkl\n- le_season.pkl")
    if st.checkbox("Show directory contents"):
        st.write(os.listdir('.'))
