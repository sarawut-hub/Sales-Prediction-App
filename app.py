# Import necessary libraries
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configuration ---
st.set_page_config(
    page_title="Sales Prediction App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better styling ---
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prediction-result {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.title("📊 Sales Prediction Dashboard")
st.markdown("### Predict sales based on social media engagement")
st.markdown("---")

# --- Load Model ---
@st.cache_resource
def load_model():
    model_filename = 'model-reg-67130700327.pkl'
    try:
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)
        return model, True
    except FileNotFoundError:
        return None, False

loaded_model, model_loaded = load_model()

# --- Sidebar for Input Parameters ---
st.sidebar.header("🎯 Input Parameters")
st.sidebar.markdown("Adjust the social media metrics below:")

if model_loaded:
    st.sidebar.success("✅ Model loaded successfully!")
    
    # Input sliders
    youtube_views = st.sidebar.slider(
        "📺 YouTube Views (K)", 
        min_value=0, 
        max_value=1000, 
        value=50, 
        step=5,
        help="Number of YouTube views in thousands"
    )
    
    tiktok_views = st.sidebar.slider(
        "🎵 TikTok Views (K)", 
        min_value=0, 
        max_value=1000, 
        value=50, 
        step=5,
        help="Number of TikTok views in thousands"
    )
    
    instagram_views = st.sidebar.slider(
        "📷 Instagram Views (K)", 
        min_value=0, 
        max_value=1000, 
        value=50, 
        step=5,
        help="Number of Instagram views in thousands"
    )
    
    # Advanced options
    with st.sidebar.expander("⚙️ Advanced Options"):
        show_charts = st.checkbox("Show detailed charts", value=True)
        show_comparison = st.checkbox("Show platform comparison", value=True)
    
else:
    st.sidebar.error("❌ Model file not found!")
    st.sidebar.markdown("Please ensure 'model-reg-67130700327.pkl' exists in the directory.")

# --- Main Content ---
if model_loaded:
    # Create input dataframe
    input_data = pd.DataFrame({
        'youtube': [youtube_views],
        'tiktok': [tiktok_views],
        'instagram': [instagram_views]
    })
    
    # Make prediction
    predicted_sales = loaded_model.predict(input_data)[0]
    
    # --- Results Section ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📺 YouTube Views",
            value=f"{youtube_views:,}K",
            delta=f"{youtube_views - 50:+}K from baseline"
        )
    
    with col2:
        st.metric(
            label="🎵 TikTok Views", 
            value=f"{tiktok_views:,}K",
            delta=f"{tiktok_views - 50:+}K from baseline"
        )
    
    with col3:
        st.metric(
            label="📷 Instagram Views",
            value=f"{instagram_views:,}K", 
            delta=f"{instagram_views - 50:+}K from baseline"
        )
    
    # --- Prediction Result ---
    st.markdown("### 🎯 Prediction Result")
    
    # Create a beautiful prediction display
    st.markdown(f"""
    <div class="prediction-result">
        <h2>💰 Estimated Sales</h2>
        <h1>${predicted_sales:,.2f}</h1>
        <p>Based on your social media engagement metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Charts Section ---
    if show_charts:
        st.markdown("### 📈 Visual Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Platform comparison pie chart
            if show_comparison:
                platform_data = pd.DataFrame({
                    'Platform': ['YouTube', 'TikTok', 'Instagram'],
                    'Views': [youtube_views, tiktok_views, instagram_views],
                    'Colors': ['#FF0000', '#000000', '#E4405F']
                })
                
                fig_pie = px.pie(
                    platform_data, 
                    values='Views', 
                    names='Platform',
                    title="Platform Distribution",
                    color='Platform',
                    color_discrete_map={
                        'YouTube': '#FF0000',
                        'TikTok': '#000000', 
                        'Instagram': '#E4405F'
                    }
                )
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart comparison
            fig_bar = px.bar(
                x=['YouTube', 'TikTok', 'Instagram'],
                y=[youtube_views, tiktok_views, instagram_views],
                title="Views by Platform",
                labels={'x': 'Platform', 'y': 'Views (K)'},
                color=['YouTube', 'TikTok', 'Instagram'],
                color_discrete_map={
                    'YouTube': '#FF0000',
                    'TikTok': '#000000',
                    'Instagram': '#E4405F'
                }
            )
            fig_bar.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # --- Data Table ---
    with st.expander("📋 Input Data Summary"):
        st.dataframe(
            input_data.T.rename(columns={0: 'Value (K)'}),
            use_container_width=True
        )
    
    # --- Sensitivity Analysis ---
    st.markdown("### 🔍 Sensitivity Analysis")
    st.markdown("See how changes in each platform affect predicted sales:")
    
    # Create sensitivity data
    base_youtube, base_tiktok, base_instagram = youtube_views, tiktok_views, instagram_views
    sensitivity_range = np.arange(0, 201, 20)
    
    youtube_sensitivity = []
    tiktok_sensitivity = []
    instagram_sensitivity = []
    
    for val in sensitivity_range:
        # YouTube sensitivity
        temp_data = pd.DataFrame({'youtube': [val], 'tiktok': [base_tiktok], 'instagram': [base_instagram]})
        youtube_sensitivity.append(loaded_model.predict(temp_data)[0])
        
        # TikTok sensitivity
        temp_data = pd.DataFrame({'youtube': [base_youtube], 'tiktok': [val], 'instagram': [base_instagram]})
        tiktok_sensitivity.append(loaded_model.predict(temp_data)[0])
        
        # Instagram sensitivity
        temp_data = pd.DataFrame({'youtube': [base_youtube], 'tiktok': [base_tiktok], 'instagram': [val]})
        instagram_sensitivity.append(loaded_model.predict(temp_data)[0])
    
    # Create sensitivity plot
    fig_sensitivity = go.Figure()
    
    fig_sensitivity.add_trace(go.Scatter(
        x=sensitivity_range, y=youtube_sensitivity,
        mode='lines+markers', name='YouTube Impact',
        line=dict(color='#FF0000', width=3)
    ))
    
    fig_sensitivity.add_trace(go.Scatter(
        x=sensitivity_range, y=tiktok_sensitivity,
        mode='lines+markers', name='TikTok Impact',
        line=dict(color='#000000', width=3)
    ))
    
    fig_sensitivity.add_trace(go.Scatter(
        x=sensitivity_range, y=instagram_sensitivity,
        mode='lines+markers', name='Instagram Impact',
        line=dict(color='#E4405F', width=3)
    ))
    
    fig_sensitivity.update_layout(
        title="Sales Sensitivity to Platform Changes",
        xaxis_title="Views (K)",
        yaxis_title="Predicted Sales ($)",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_sensitivity, use_container_width=True)

else:
    st.error("🚫 Cannot load the prediction model. Please check if the model file exists.")
    st.info("💡 Make sure 'model-reg-67130700327.pkl' is in the same directory as this app.")

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "📊 Sales Prediction Dashboard | Built with Streamlit"
    "</div>", 
    unsafe_allow_html=True
)
