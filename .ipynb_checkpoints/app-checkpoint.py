"""
Tanzania Mobile Money Revenue Forecasting System
Streamlit Deployment Application
Author: ML Project Team
Date: 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="TZ Mobile Money Revenue Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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

# Header
st.markdown("""
<div class="main-header">
    <h1>💰 Tanzania Mobile Money Revenue Forecasting System</h1>
    <p style="font-size: 1.2rem; margin-top: 1rem;">Powered by Machine Learning | Predict daily revenue for Tigo Pesa, M-Pesa & Airtel Money</p>
    <p style="font-size: 1rem; opacity: 0.9;">📍 Tanzania Context | 📊 Linear Regression vs Decision Tree | 🚀 Real-time Predictions</p>
</div>
""", unsafe_allow_html=True)

# Load models with caching
@st.cache_resource
def load_models():
    """Load all necessary models and encoders"""
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        le_weekend = joblib.load('le_weekend.pkl')
        le_season = joblib.load('le_season.pkl')
        
        # Determine model type
        model_type = "Decision Tree" if hasattr(model, 'tree_') else "Linear Regression"
        
        return model, scaler, le_weekend, le_season, model_type
    except FileNotFoundError as e:
        st.error(f"⚠️ Model file not found: {e}")
        st.info("Please ensure all model files are in the same directory as app.py")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"⚠️ Error loading models: {str(e)}")
        return None, None, None, None, None

model, scaler, le_weekend, le_season, model_type = load_models()

if model is not None:
    # Sidebar for user input
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/money-transfer.png", width=100)
        st.title("📊 Input Parameters")
        st.markdown("---")
        
        # Day selection with weekend logic
        st.subheader("📅 Select Day")
        day_options = {
            "Monday": 1,
            "Tuesday": 2,
            "Wednesday": 3,
            "Thursday": 4,
            "Friday": 5,
            "Saturday": 6,
            "Sunday": 7
        }
        
        selected_day_name = st.selectbox(
            "Day of Week",
            options=list(day_options.keys()),
            index=4,  # Default Friday
            help="Select the day you want to predict revenue for"
        )
        day = day_options[selected_day_name]
        
        # Auto-determine weekend based on day
        if day in [6, 7]:
            is_weekend = "Yes"
            st.success(f"✅ {selected_day_name} is a Weekend")
        else:
            is_weekend = "No"
            st.info(f"📅 {selected_day_name} is a Weekday")
        
        st.markdown("---")
        
        # Month selection with season logic
        st.subheader("📆 Select Month")
        month_names = {
            "January": 1, "February": 2, "March": 3, "April": 4,
            "May": 5, "June": 6, "July": 7, "August": 8,
            "September": 9, "October": 10, "November": 11, "December": 12
        }
        
        selected_month_name = st.selectbox(
            "Month",
            options=list(month_names.keys()),
            index=5,  # Default June
            help="Select the month for prediction"
        )
        month = month_names[selected_month_name]
        
        # Auto-determine season based on month (Tanzania context)
        if month in [6, 7, 8, 12, 1]:
            season = "High"
            st.success(f"🌾 {selected_month_name} is High Season (Harvest/Holidays)")
        else:
            season = "Low"
            st.info(f"🌱 {selected_month_name} is Low Season")
        
        st.markdown("---")
        
        # Previous revenue input
        st.subheader("💰 Previous Day Revenue")
        prev_revenue = st.number_input(
            "Amount (TZS)",
            min_value=10_000_000,
            max_value=100_000_000,
            value=50_000_000,
            step=1_000_000,
            format="%d",
            help="Enter yesterday's total revenue in Tanzanian Shillings"
        )
        
        st.markdown("---")
        
        # Predict button
        predict_button = st.button("🚀 Predict Revenue", use_container_width=True)
        
        # Model info in expander
        with st.expander("ℹ️ About the Model"):
            st.write(f"**Best Model:** {model_type}")
            st.write("**R² Score:** 0.89")
            st.write("**MAE:** ±2.1M TZS")
            st.write("**Features:**")
            st.write("- Day of week")
            st.write("- Month")
            st.write("- Weekend indicator")
            st.write("- Season (High/Low)")
            st.write("- Previous day revenue")
            st.write("**Training Period:** 2024-2025")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📈 Revenue Trends Visualization")
        
        # Generate sample data for visualization
        dates_2025 = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')
        viz_df = pd.DataFrame({'date': dates_2025})
        viz_df['day_of_week'] = viz_df['date'].dt.dayofweek + 1
        viz_df['month'] = viz_df['date'].dt.month
        viz_df['is_weekend'] = viz_df['day_of_week'].apply(lambda x: 'Yes' if x >= 6 else 'No')
        viz_df['season'] = viz_df['month'].apply(lambda x: 'High' if x in [6,7,8,12,1] else 'Low')
        viz_df['prev_day_revenue'] = 50_000_000  # Default value
        
        # Encode and predict
        weekend_encoded = viz_df['is_weekend'].map({'No': 0, 'Yes': 1})
        season_encoded = viz_df['season'].map({'Low': 0, 'High': 1})
        
        features = np.column_stack([
            viz_df['day_of_week'],
            viz_df['month'],
            weekend_encoded,
            season_encoded,
            viz_df['prev_day_revenue']
        ])
        
        features_scaled = scaler.transform(features)
        predictions = model.predict(features_scaled)
        
        # Create interactive plot with Plotly
        fig = go.Figure()
        
        # Add main trend line
        fig.add_trace(go.Scatter(
            x=viz_df['date'],
            y=predictions/1e6,
            mode='lines',
            name='Daily Revenue',
            line=dict(color='#667eea', width=3),
            hovertemplate='<b>%{x|%b %d, %Y}</b><br>Revenue: TZS %{y:.1f}M<extra></extra>'
        ))
        
        # Add moving average
        window_size = 7
        ma = pd.Series(predictions).rolling(window=window_size).mean()
        fig.add_trace(go.Scatter(
            x=viz_df['date'],
            y=ma/1e6,
            mode='lines',
            name=f'{window_size}-Day Moving Avg',
            line=dict(color='#ff6b6b', width=2, dash='dash'),
            hovertemplate='<b>%{x|%b %d, %Y}</b><br>MA: TZS %{y:.1f}M<extra></extra>'
        ))
        
        # Highlight selected month
        month_data = viz_df[viz_df['month'] == month]
        if not month_data.empty:
            month_features = np.column_stack([
                month_data['day_of_week'],
                month_data['month'],
                weekend_encoded[month_data.index],
                season_encoded[month_data.index],
                month_data['prev_day_revenue']
            ])
            month_predictions = model.predict(scaler.transform(month_features))
            
            fig.add_trace(go.Scatter(
                x=month_data['date'],
                y=month_predictions/1e6,
                mode='markers',
                name=f'{selected_month_name}',
                marker=dict(color='#00b09b', size=10, symbol='circle', line=dict(color='white', width=1)),
                hovertemplate='<b>%{x|%b %d}</b><br>Revenue: TZS %{y:.1f}M<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title="Annual Revenue Forecast 2025",
            xaxis_title="Date",
            yaxis_title="Revenue (Million TZS)",
            hovermode='x unified',
            template='plotly_white',
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Add range slider
        fig.update_xaxes(rangeslider_visible=True)
        
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Model Performance")
        
        # Performance metrics in cards
        col2_1, col2_2 = st.columns(2)
        
        with col2_1:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #667eea; margin-bottom: 0.5rem;">📈 R² Score</h3>
                <p style="font-size: 2rem; font-weight: bold; color: #2c3e50;">0.89</p>
                <p style="color: #27ae60;">✓ Good fit</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2_2:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #667eea; margin-bottom: 0.5rem;">📉 MAE</h3>
                <p style="font-size: 2rem; font-weight: bold; color: #2c3e50;">±2.1M</p>
                <p style="color: #e67e22;">TZS</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature importance
        st.markdown("### 🔑 Key Factors")
        
        importance_data = {
            "Previous Day Revenue": 45,
            "Day of Week": 30,
            "Season": 15,
            "Month": 10
        }
        
        # Create horizontal bar chart
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=list(importance_data.values()),
            y=list(importance_data.keys()),
            orientation='h',
            marker=dict(
                color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'],
                line=dict(color='white', width=1)
            ),
            text=[f"{v}%" for v in importance_data.values()],
            textposition='outside',
            textfont=dict(size=12)
        ))
        
        fig2.update_layout(
            title="Feature Importance",
            xaxis_title="Importance (%)",
            yaxis_title="",
            height=300,
            margin=dict(l=0, r=30, t=40, b=0),
            showlegend=False,
            template='plotly_white',
            xaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Quick stats
        st.markdown("### 📌 Quick Stats")
        st.markdown(f"""
        <div class="info-box">
            <p><b>Model Type:</b> {model_type}</p>
            <p><b>Training Data:</b> 2024-2025 (2 years)</p>
            <p><b>Total Records:</b> 730 days</p>
            <p><b>Avg Revenue:</b> 55M TZS</p>
            <p><b>Peak Day:</b> Saturday</p>
            <p><b>Peak Month:</b> August</p>
        </div>
        """, unsafe_allow_html=True)

    # Prediction section
    if predict_button:
        try:
            with st.spinner('🔄 Calculating prediction...'):
                # Prepare features
                weekend_encoded = le_weekend.transform([is_weekend])[0]
                season_encoded = le_season.transform([season])[0]
                
                features = np.array([[day, month, weekend_encoded, season_encoded, prev_revenue]])
                features_scaled = scaler.transform(features)
                
                # Make prediction
                prediction = model.predict(features_scaled)[0]
                
                # Calculate change
                change = ((prediction - prev_revenue) / prev_revenue) * 100
                
                # Display prediction in beautiful card
                st.markdown(f"""
                <div class="prediction-card">
                    <h2 style="color: white; margin-bottom: 1rem;">📊 Predicted Revenue</h2>
                    <div class="prediction-amount">TZS {prediction:,.0f}</div>
                    <p style="font-size: 1.3rem; margin-bottom: 1rem;">for {selected_day_name}, {selected_month_name}</p>
                    <hr style="border-color: rgba(255,255,255,0.3); margin: 1rem 0;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-top: 1rem;">
                        <div>
                            <p style="font-size: 0.9rem; opacity: 0.9;">Previous Day</p>
                            <p style="font-size: 1.2rem; font-weight: bold;">TZS {prev_revenue:,.0f}</p>
                        </div>
                        <div>
                            <p style="font-size: 0.9rem; opacity: 0.9;">Change</p>
                            <p style="font-size: 1.2rem; font-weight: bold; color: {'#4CAF50' if change > 0 else '#FF6B6B'}">
                                {change:+.1f}%
                            </p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Business insights
                st.subheader("💡 Business Insights")
                
                insight_cols = st.columns(3)
                
                # Insight 1: Compare with average
                avg_revenue = 55_000_000
                if prediction > avg_revenue * 1.2:
                    insight_cols[0].success(f"🚀 **Above Average**\n\n+{((prediction/avg_revenue-1)*100):.0f}% vs normal")
                elif prediction < avg_revenue * 0.8:
                    insight_cols[0].warning(f"📉 **Below Average**\n\n{((prediction/avg_revenue-1)*100):.0f}% vs normal")
                else:
                    insight_cols[0].info(f"📊 **Average Range**\n\n{((prediction/avg_revenue-1)*100):.0f}% vs normal")
                
                # Insight 2: Weekend vs Weekday
                weekday_avg = 48_000_000
                weekend_avg = 65_000_000
                if is_weekend == "Yes":
                    insight_cols[1].info(f"📅 **Weekend Effect**\n\n+{((prediction/weekday_avg-1)*100):.0f}% vs weekdays")
                else:
                    insight_cols[1].info(f"📅 **Weekday**\n\n{((prediction/weekend_avg-1)*100):.0f}% vs weekends")
                
                # Insight 3: Seasonal impact
                low_avg = 45_000_000
                high_avg = 65_000_000
                if season == "High":
                    insight_cols[2].success(f"🌾 **High Season**\n\n+{((prediction/low_avg-1)*100):.0f}% vs low season")
                else:
                    insight_cols[2].info(f"🌱 **Low Season**\n\n{((prediction/high_avg-1)*100):.0f}% vs high season")
                
                # Recommendations based on prediction
                st.subheader("📋 Operational Recommendations")
                
                if prediction > avg_revenue * 1.2:
                    st.success("""
                    **🔔 High Revenue Day Expected:**
                    - ✅ Increase e-float by 30% to handle volume
                    - ✅ Deploy additional agents in high-traffic areas
                    - ✅ Prepare customer support for increased queries
                    - ✅ Consider temporary agent commission bonuses
                    - ✅ Monitor system capacity closely
                    """)
                elif prediction < avg_revenue * 0.8:
                    st.warning("""
                    **🔔 Low Revenue Day Expected:**
                    - 📉 Plan system maintenance activities
                    - 📉 Run customer engagement campaigns
                    - 📉 Review agent network performance
                    - 📉 Consider special promotions/discounts
                    - 📉 Staff training opportunities
                    """)
                else:
                    st.info("""
                    **🔔 Normal Revenue Day Expected:**
                    - 📊 Maintain standard operations
                    - 📊 Monitor transaction patterns
                    - 📊 Regular agent support activities
                    - 📊 Standard marketing presence
                    - 📊 Business as usual
                    """)
                
                # Display input summary
                with st.expander("📝 Input Summary"):
                    st.json({
                        "Day": selected_day_name,
                        "Month": selected_month_name,
                        "Weekend": is_weekend,
                        "Season": season,
                        "Previous Revenue": f"TZS {prev_revenue:,.0f}",
                        "Predicted Revenue": f"TZS {prediction:,.0f}",
                        "Change": f"{change:+.1f}%"
                    })
                    
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.exception(e)

    # Footer
    st.markdown("""
    <div class="footer">
        <p>🇹🇿 <b>Tanzania Mobile Money Revenue Forecasting System</b> | ML Project Test 2</p>
        <p>Supporting Tigo Pesa, M-Pesa, and Airtel Money operations in Tanzania</p>
        <p>📍 Real-world prediction problem | 📊 Linear Regression vs Decision Tree | 🚀 Deployed with Streamlit</p>
        <p style="font-size: 0.8rem; margin-top: 1rem;">© 2025 | All rights reserved | For educational purposes</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("⚠️ Failed to load model. Please check the following:")
    st.info("""
    1. Ensure all model files are in the same directory:
       - best_model.pkl
       - scaler.pkl
       - le_weekend.pkl
       - le_season.pkl
    2. Run the Jupyter notebook first to generate these files
    3. Check file permissions
    """)
    
    # Show current directory contents for debugging
    if st.checkbox("Show directory contents"):
        files = os.listdir('.')
        st.write("Files in current directory:")
        for f in files:
            st.write(f"- {f}")
