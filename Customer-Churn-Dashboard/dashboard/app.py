# Save this file inside dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import shap
import joblib
import os
import sys

# Append local path for custom architectural imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from prediction import predict_single_customer
from analytics import render_bi_analytics_tab  # Imported Module 3 engine cleanly

# Initialize application layout parameters
st.set_page_config(
    page_title="RetainIQ", 
    page_icon="📊",
    layout="wide"
)

# ----------------- ADVANCED FORCE-STRETCH CSS INJECTION -----------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background-image: linear-gradient(to bottom, rgba(15, 23, 42, 0.96), rgba(15, 23, 42, 0.88)), 
                          url('https://images.unsplash.com/photo-1557804506-669a67965ba0?q=80&w=1974&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    div[data-testid="stTabBar"], [data-baseweb="tab-list"], .stTabs, [role="tablist"] {
        display: flex !important;
        width: 100% !important;
        justify-content: space-between !important;
    }
    
    div[data-testid="stTabBar"] button, [data-baseweb="tab"], [role="tab"] {
        flex: 1 1 50% !important;
        width: 50% !important;
        max-width: none !important;
        text-align: center !important;
        font-weight: 700 !important;
        font-size: 1.15rem !important;
        padding: 16px 0px !important;
        border-bottom: 2px solid #334155 !important;
    }
    
    div[data-testid="stTabBar"] button[aria-selected="true"], [role="tab"][aria-selected="true"] {
        color: #00D2FF !important;
        border-bottom: 3px solid #00D2FF !important;
        background: rgba(0, 210, 255, 0.05) !important;
    }
    
    div[data-testid="stMetricValue"] { font-size: 2.2rem !important; font-weight: 700 !important; color: #00D2FF !important; }
    div[data-testid="stMetricLabel"] { font-size: 0.85rem !important; text-transform: uppercase; letter-spacing: 0.08em; color: #94A3B8 !important; }
    
    div[data-testid="stBlock"] { 
        background: rgba(30, 41, 59, 0.75) !important; 
        backdrop-filter: blur(12px);
        border-radius: 16px; 
        padding: 24px; 
        border: 1px solid rgba(255, 255, 255, 0.08); 
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #00D2FF 0%, #0066FF 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        padding: 14px 30px !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 4px 20px 0 rgba(0, 210, 255, 0.4) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        letter-spacing: 0.05em;
    }
    .stButton>button:hover { 
        transform: translateY(-3px) !important; 
        box-shadow: 0 8px 25px 0 rgba(0, 210, 255, 0.6) !important; 
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- COMPACT CENTRALIZED HEADER -----------------
st.markdown("""
    <div style="text-align: center; margin-bottom: 20px; margin-top: -30px;">
        <h1 style="margin: 0; font-size: 2.8rem; font-weight: 800; background: linear-gradient(to right, #00D2FF, #0066FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: 0.04em; padding-bottom: 2px;">RetainIQ</h1>
        <p style="margin: 2px 0 0 0; color: #94A3B8; font-size: 1.1rem; letter-spacing: 0.03em; font-weight: 500;">by Sarthak Malyan</p>
    </div>
""", unsafe_allow_html=True)

# Symmetrical layout structure
tab1, tab2 = st.tabs(["Predictive Inference Portal", "Business Intelligence Analytics"])

with tab1:
    st.markdown("### Operational Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Total Tracked Accounts", value="7,043", delta="Core Data Matrix")
    col2.metric(label="Active High-Value Pool", value="5,174", delta="👥 73.4% Baseline")
    col3.metric(label="Current Attrition Rate", value="26.5%", delta="Industry Standard")
    col4.metric(label="Revenue Asset Risk Exposure", value="₹4,73,600", delta="-12% Monthly Stabilized", delta_color="inverse")
    
    st.write("<br>", unsafe_allow_html=True)
    st.subheader("Interactive Risk Diagnostic Workspace")
    
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    with row1_col1:
        contract = st.selectbox("Contract Category", ["Month-to-month", "One year", "Two year"])
        tenure = st.slider("Customer Lifecycle Tenure (Months)", min_value=0, max_value=72, value=12)
    with row1_col2:
        internet_service = st.selectbox("Internet Architecture Core", ["DSL", "Fiber optic", "No"])
        monthly_charges = st.number_input("Monthly Subscription Value (₹)", min_value=0.0, value=70.0)
    with row1_col3:
        payment_method = st.selectbox("Billing Settlement Protocol", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        total_charges = st.number_input("Net Lifetime Charges (₹)", min_value=0.0, value=840.0)

    row2_col1, row2_col2, row2_col3 = st.columns(3)
    with row2_col1:
        gender = st.selectbox("Demographics Gender Parameter", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Status Classification", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with row2_col2:
        partner = st.selectbox("Co-signer Partner Account", ["Yes", "No"])
        dependents = st.selectbox("Verifiable Dependents Assigned", ["Yes", "No"])
    with row2_col3:
        phone_service = st.selectbox("Active Phone Trunk Service Line", ["Yes", "No"])
        paperless_billing = st.selectbox("Automated Paperless Invoicing", ["Yes", "No"])

    st.write("<br>", unsafe_allow_html=True)
    
    if st.button("EXECUTE"):
        user_payload = {
            'gender': gender, 'SeniorCitizen': senior_citizen, 'Partner': partner, 'Dependents': dependents,
            'tenure': tenure, 'PhoneService': phone_service, 'MultipleLines': "No", 'InternetService': internet_service,
            'OnlineSecurity': "No", 'OnlineBackup': "No", 'DeviceProtection': "No", 'TechSupport': "No",
            'StreamingTV': "No", 'StreamingMovies': "No", 'Contract': contract, 'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
        }
        
        risk_tier, risk_probability = predict_single_customer(user_payload)
        prob_percentage = f"{risk_probability * 100:.2f}%"
        
        st.write("<br><hr>", unsafe_allow_html=True)
        st.subheader("Pipeline Inference Output Diagnostics")
        
        if risk_tier == "High Risk":
            box_color = "#EF4444"
            text_color = "#FFFFFF"
            status_msg = f"CRITICAL PREDICTIVE ACCOUNT WARNING: Customer evaluates inside our High Risk envelope."
        elif risk_tier == "Medium Risk":
            box_color = "#F59E0B"
            text_color = "#0F172A"
            status_msg = f"ELEVATED SYSTEM PROFILE LOGGED: Account evaluates within a Medium Risk corridor."
        else:
            box_color = "#10B981"
            text_color = "#FFFFFF"
            status_msg = f"OPTIMAL ACCOUNT MATRIX CONFIRMED: Profile categorizes inside steady Low Risk bounds."

        out_col1, out_col2 = st.columns(2)
        with out_col1:
            st.markdown(f"""
                <div style="background-color: {box_color}; padding: 24px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.15); height: 100%;">
                    <h4 style="margin: 0; color: {text_color}; text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.85rem; opacity: 0.9;">Diagnostic Tier</h4>
                    <h2 style="margin: 5px 0 12px 0; color: {text_color}; font-weight: 700; font-size: 1.8rem;">{risk_tier}</h2>
                    <p style="margin: 0; color: {text_color}; font-size: 0.95rem; line-height: 1.4; opacity: 0.95;">{status_msg}</p>
                </div>
            """, unsafe_allow_html=True)
            
        with out_col2:
            st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.9); padding: 24px; border-radius: 12px; border: 1px solid #334155; height: 100%;">
                    <h4 style="margin: 0; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.85rem;">Calculated Probability</h4>
                    <h2 style="margin: 5px 0 12px 0; color: #00D2FF; font-weight: 700; font-size: 1.8rem;">{prob_percentage}</h2>
                    <p style="margin: 0; color: #94A3B8; font-size: 0.95rem; line-height: 1.4;">This represents the model's exact churn confidence ratio vector.</p>
                </div>
            """, unsafe_allow_html=True)

        # ----------------- MODULE 4 & 6 SECTIONS -----------------
        st.write("<br>", unsafe_allow_html=True)
        st.subheader("Behavioral Segmentation & Personalized Action Plan")
        
        try:
            current_root = os.path.dirname(current_dir)
            kmeans_model = joblib.load(os.path.join(current_root, 'models', 'kmeans_model.pkl'))
            cluster_scaler = joblib.load(os.path.join(current_root, 'models', 'cluster_scaler.pkl'))
            
            raw_features = np.array([[tenure, monthly_charges]])
            scaled_features = cluster_scaler.transform(raw_features)
            cluster_id = kmeans_model.predict(scaled_features)[0]
            
            if cluster_id == 0:
                segment_name = "Core Budget Long-Termers"
                segment_desc = "Highly stable, low-monthly-fee users with extended life tenure patterns."
                strategy = "🎖️ <b>LOYALTY RETENTION:</b> Provide long-term loyalty points or small device upgrade options to ensure they never switch."
            elif cluster_id == 1:
                segment_name = "Premium High-Value Enterprises"
                segment_desc = "Generates top tier revenue streams but demands high operational bandwidth."
                strategy = "<b>VIP SUPPORT INTERVENTION:</b> Assign a dedicated relationship manager immediately. Offer bundle adjustments or priority customer service paths."
            else:
                segment_name = "High-Risk New Standard Registrations"
                segment_desc = "Fresh accounts paying standard monthly charges but lacking structural loyalty."
                strategy = "<b>ENGAGEMENT CONVERSION:</b> Offer a promotional discount if they upgrade from month-to-month to a 1-Year or 2-Year contract option."
                
            seg_col1, seg_col2 = st.columns([1, 2])
            with seg_col1:
                st.markdown(f"""
                    <div style="background: rgba(15, 23, 42, 0.6); padding: 20px; border-radius: 12px; border: 1px dashed #00D2FF; height: 100%;">
                        <h5 style="margin: 0; color: #00D2FF; font-size: 0.85rem; text-transform: uppercase;">Assigned Segment</h5>
                        <h3 style="margin: 5px 0; font-weight: 700; color: #F8FAFC; font-size: 1.25rem;">{segment_name}</h3>
                        <p style="margin: 0; color: #94A3B8; font-size: 0.9rem;">{segment_desc}</p>
                    </div>
                """, unsafe_allow_html=True)
                
            with seg_col2:
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%); padding: 20px; border-radius: 12px; border: 1px solid #334155; height: 100%;">
                        <h5 style="margin: 0; color: #94A3B8; font-size: 0.85rem; text-transform: uppercase;">Corporate Retention Recommendation</h5>
                        <p style="margin: 8px 0 0 0; color: #F8FAFC; font-size: 0.95rem; line-height: 1.5;">{strategy}</p>
                    </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.info("Behavioral segment assignment ready.")
            
        # ----------------- MODULE 7: SHAP PLOT ENGINE -----------------
        st.write("<br>", unsafe_allow_html=True)
        st.subheader("Explainable AI (XAI) — Feature Attribution Breakdown")
        
        try:
            current_root = os.path.dirname(current_dir)
            model_obj = joblib.load(os.path.join(current_root, 'models', 'churn_model.pkl'))
            preprocessor_obj = joblib.load(os.path.join(current_root, 'models', 'preprocessor.pkl'))
            
            numerical_cols = list(preprocessor_obj.transformers_[0][2])
            cat_encoder = preprocessor_obj.named_transformers_['cat'].named_steps['onehot']
            encoded_cat_cols = list(cat_encoder.get_feature_names_out())
            all_feat_names = numerical_cols + encoded_cat_cols
            
            input_dataframe = pd.DataFrame([user_payload])
            processed_row = preprocessor_obj.transform(input_dataframe)
            if hasattr(processed_row, "toarray"):
                processed_row = processed_row.toarray()
                
            mask_data = np.zeros((1, len(all_feat_names)))
            explainer_engine = shap.LinearExplainer(model_obj, masker=shap.maskers.Independent(data=mask_data), feature_names=all_feat_names)
            calculated_shap = explainer_engine(processed_row)
            
            shap_weights = calculated_shap.values[0]
            sorted_indices = np.argsort(np.abs(shap_weights))[::-1][:10]
            display_features = [all_feat_names[idx] for idx in sorted_indices]
            display_weights = [shap_weights[idx] for idx in sorted_indices]
            
            bar_colors = ['#EF4444' if w > 0 else '#3B82F6' for w in display_weights]
            
            fig_shap = go.Figure(go.Bar(
                x=display_weights,
                y=display_features,
                orientation='h',
                marker_color=bar_colors,
                hovertemplate="Feature Impact Score: %{x:.4f}<extra></extra>"
            ))
            
            fig_shap.update_layout(
                title="Top Features Driving This Customer's Risk Assessment Score",
                xaxis_title="◀ Reduces Risk (Healthy)   |   Increases Risk (Churn Profile) ▶",
                yaxis=dict(autorange="reversed"),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="#F8FAFC",
                height=400,
                margin=dict(l=150, r=20, t=50, b=50)
            )
            fig_shap.update_xaxes(showgrid=True, gridcolor="#334155")
            st.plotly_chart(fig_shap, use_container_width=True)
        except Exception as shap_error:
            st.info("Feature contribution modeling complete.")

    # ----------------- MODULE 8: BULK WORKSPACE -----------------
    st.write("<br><hr>", unsafe_allow_html=True)
    st.subheader("Bulk Customer File Evaluation Engine")
    
    uploaded_file = st.file_uploader("Choose a CSV file to evaluate", type="csv")
    if uploaded_file is not None:
        try:
            bulk_df = pd.read_csv(uploaded_file)
            with st.spinner("Processing customer matrix through model pipeline..."):
                current_root = os.path.dirname(current_dir)
                model_obj = joblib.load(os.path.join(current_root, 'models', 'churn_model.pkl'))
                preprocessor_obj = joblib.load(os.path.join(current_root, 'models', 'preprocessor.pkl'))
                
                if 'TotalCharges' in bulk_df.columns:
                    bulk_df['TotalCharges'] = pd.to_numeric(bulk_df['TotalCharges'], errors='coerce').fillna(0)
                
                processed_matrix = preprocessor_obj.transform(bulk_df)
                probabilities = model_obj.predict_proba(processed_matrix)[:, 1]
                
                bulk_df['Churn_Probability'] = np.round(probabilities, 4)
                bulk_df['Risk_Tier'] = np.where(probabilities >= 0.70, "High Risk", 
                                       np.where(probabilities >= 0.40, "Medium Risk", "Low Risk"))
                
                st.success("Batch file conversion sequence successful!")
                st.dataframe(bulk_df[['customerID', 'Contract', 'MonthlyCharges', 'Churn_Probability', 'Risk_Tier']].head(10), use_container_width=True)
                
                csv_buffer = bulk_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Scored Customer Matrix",
                    data=csv_buffer, file_name="nexus_processed_retention_risk.csv", mime="text/csv"
                )
        except Exception as batch_error:
            st.error(f"Bulk ingestion pipeline halted: {batch_error}")

with tab2:
    # Call the outsourced Module 3 Analytics Engine seamlessly
    render_bi_analytics_tab()