# Save this file inside dashboard/analytics.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os

@st.cache_data
def load_unified_analytics_data():
    """
    Foolproof absolute path loader designed to fetch the master dataset
    reliably across local and cloud server environments.
    """
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)
    data_path = os.path.join(project_root, 'data', 'telco_churn.csv')
    
    df = pd.read_csv(data_path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    return df

def render_bi_analytics_tab():
    """
    Renders the complete interactive Business Intelligence dashboard core.
    """
    st.markdown("### 📈 Historical Customer Trends Insights Platform")
    
    try:
        analytics_df = load_unified_analytics_data()
        
        # 1. Attrition by Contract Type Chart
        fig_contract = px.histogram(
            analytics_df, x="Contract", color="Churn", barmode="group",
            title="Attrition Vectors by Contract Type", 
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        # 2. Tenure Box Plot Chart
        fig_tenure = px.box(
            analytics_df, x="Churn", y="tenure", 
            title="Customer Lifespan Tenure Distribution vs Attrition",
            color="Churn", color_discrete_sequence=['#10B981', '#EF4444']
        )
        
        # 3. Monthly Financial Commitments Spread Chart
        fig_charges = px.violin(
            analytics_df, x="Churn", y="MonthlyCharges", box=True, 
            title="Monthly Financial Commitments Spread",
            color="Churn", color_discrete_sequence=['#3B82F6', '#F59E0B']
        )
        
        # Apply unified premium UI styles across all active canvas frames
        for fig in [fig_contract, fig_tenure, fig_charges]:
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                font_color="#F8FAFC"
            )
            
        # Draw charts smoothly layout grids
        st.plotly_chart(fig_contract, use_container_width=True)
        
        col_graph1, col_graph2 = st.columns(2)
        with col_graph1:
            st.plotly_chart(fig_tenure, use_container_width=True)
        with col_graph2:
            st.plotly_chart(fig_charges, use_container_width=True)
            
    except Exception as e:
        st.error(f"Could not render business data visualizations: {e}")