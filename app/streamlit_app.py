import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st

from src.inference import get_predictor
from src.utils import get_project_root

# Page Configuration
st.set_page_config(
    page_title="Telecom Customer Churn Predictor & Retention System",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .badge-high {
        background-color: #FEE2E2;
        color: #DC2626;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #FCA5A5;
    }
    .badge-medium {
        background-color: #FEF3C7;
        color: #D97706;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #FCD34D;
    }
    .badge-low {
        background-color: #D1FAE5;
        color: #059669;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #6EE7B7;
    }
    .rec-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)



# --- Helper: Query Prediction via FastAPI or In-Memory fallback ---
def run_prediction(payload: dict, api_url: str) -> dict:
    # Try calling the FastAPI endpoint first
    try:
        clean_url = api_url.rstrip("/")
        predict_endpoint = f"{clean_url}/predict"
        resp = requests.post(predict_endpoint, json=payload, timeout=3.0)
        if resp.status_code == 200:
            return resp.json(), "REST API (FastAPI)"
    except Exception:
        pass

    # Direct fallback to in-memory inference engine
    predictor = get_predictor()
    res = predictor.predict(payload)
    return res, "In-Memory Engine (Fallback)"


def check_api_health(api_url: str):
    try:
        clean_url = api_url.rstrip("/")
        health_endpoint = f"{clean_url}/health"
        resp = requests.get(health_endpoint, timeout=1.5)
        if resp.status_code == 200:
            return True, resp.json()
    except Exception:
        pass
    return False, None


# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/satellite-sending-signal.png", width=64)
    st.title("System Controls")
    
    default_api_url = os.getenv("API_URL", os.getenv("FASTAPI_URL", "http://localhost:8000"))
    api_base_url = st.text_input("FastAPI Base URL", value=default_api_url)
    is_healthy, health_data = check_api_health(api_base_url)
    
    if is_healthy:
        st.success(f"🟢 API Connected (v{health_data.get('model_version', '1.0.0')})")
        st.caption(f"Uptime: {health_data.get('uptime_seconds', 0):.1f}s")
    else:
        st.warning("🟠 API Offline (Using In-Memory Fallback)")


    st.markdown("---")
    st.subheader("📊 Production Model Info")
    st.caption("**Model Family:** LightGBM Classifier")
    st.caption("**Tuned Metric:** Recall = 79.89% | ROC-AUC = 0.8415")
    st.caption("**Features Evaluated:** 28 Engineered Features")
    st.caption("**Deployment:** FastAPI REST + Streamlit UI")


# --- MAIN HEADER ---
st.markdown('<div class="main-title">Telecom Customer Churn Prediction & Retention System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-time risk scoring, predictive probability assessment, and automated retention recommendations for customer success teams.</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🎯 Single Customer Risk Assessment", 
    "📁 Batch Customer Prediction", 
    "📈 Model Benchmark & Metrics"
])

# =========================================================================
# TAB 1: SINGLE CUSTOMER PREDICTION
# =========================================================================
with tab1:
    defaults = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 12.0, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 79.5, "TotalCharges": 954.0
    }

    with st.form("customer_prediction_form"):
        st.subheader("Customer Profile & Subscription Parameters")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 1. Demographics")
            gender = st.selectbox("Gender", options=["Female", "Male"], index=0 if defaults["gender"] == "Female" else 1)
            senior_citizen = st.selectbox("Senior Citizen", options=[0, 1], format_func=lambda x: "Yes (>= 65)" if x == 1 else "No", index=defaults["SeniorCitizen"])
            partner = st.selectbox("Has Partner", options=["Yes", "No"], index=0 if defaults["Partner"] == "Yes" else 1)
            dependents = st.selectbox("Has Dependents", options=["Yes", "No"], index=0 if defaults["Dependents"] == "Yes" else 1)

        with col2:
            st.markdown("##### 🌐 2. Subscribed Services")
            phone_service = st.selectbox("Phone Service", options=["Yes", "No"], index=0 if defaults["PhoneService"] == "Yes" else 1)
            multiple_lines = st.selectbox("Multiple Lines", options=["No", "Yes", "No phone service"], index=["No", "Yes", "No phone service"].index(defaults["MultipleLines"]))
            internet_service = st.selectbox("Internet Service", options=["DSL", "Fiber optic", "No"], index=["DSL", "Fiber optic", "No"].index(defaults["InternetService"]))
            online_security = st.selectbox("Online Security", options=["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["OnlineSecurity"]))
            online_backup = st.selectbox("Online Backup", options=["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["OnlineBackup"]))
            device_protection = st.selectbox("Device Protection", options=["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["DeviceProtection"]))
            tech_support = st.selectbox("Tech Support", options=["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["TechSupport"]))
            streaming_tv = st.selectbox("Streaming TV", options=["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["StreamingTV"]))
            streaming_movies = st.selectbox("Streaming Movies", options=["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["StreamingMovies"]))

        with col3:
            st.markdown("##### 💳 3. Account & Billing Terms")
            tenure = st.slider("Tenure (Months with Company)", min_value=0.0, max_value=72.0, value=float(defaults["tenure"]), step=1.0)
            contract = st.selectbox("Contract Type", options=["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(defaults["Contract"]))
            paperless_billing = st.selectbox("Paperless Billing", options=["Yes", "No"], index=0 if defaults["PaperlessBilling"] == "Yes" else 1)
            payment_method = st.selectbox(
                "Payment Method", 
                options=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(defaults["PaymentMethod"])
            )
            monthly_charges = st.slider("Monthly Charges ($ USD)", min_value=18.0, max_value=125.0, value=float(defaults["MonthlyCharges"]), step=0.5)
            
            # Default TotalCharges calculation
            calc_total = round(float(defaults.get("TotalCharges", tenure * monthly_charges)), 2)
            total_charges = st.number_input("Total Charges ($ USD)", min_value=0.0, max_value=10000.0, value=calc_total, step=50.0)

        submit_btn = st.form_submit_button("🚀 Predict Customer Churn Risk", type="primary", use_container_width=True)

    if submit_btn:
        payload = {
            "customerID": "WEB-CUST-USER",
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }

        with st.spinner("Analyzing customer risk profile..."):
            res, engine_source = run_prediction(payload, api_base_url)

        st.markdown("---")
        st.subheader("Prediction Results & Risk Assessment")

        prob = res["churn_probability"]
        prob_pct = res["risk_score_percent"]
        risk_lvl = res["risk_level"]
        is_churn = res["churn_prediction"] == 1

        # KPI Metrics Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        with kpi1:
            badge_class = "badge-high" if risk_lvl == "High" else ("badge-medium" if risk_lvl == "Medium" else "badge-low")
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #64748B; font-size: 0.85rem; font-weight: 600;">RISK CLASSIFICATION</div>
                <div style="margin-top: 10px;"><span class="{badge_class}">{risk_lvl.upper()} RISK</span></div>
            </div>
            """, unsafe_allow_html=True)

        with kpi2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #64748B; font-size: 0.85rem; font-weight: 600;">CHURN PROBABILITY</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #1E293B; margin-top: 5px;">{prob_pct}%</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi3:
            decision_text = "LIKELY TO CHURN" if is_churn else "LIKELY TO STAY"
            decision_color = "#DC2626" if is_churn else "#059669"
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #64748B; font-size: 0.85rem; font-weight: 600;">MODEL PREDICTION</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: {decision_color}; margin-top: 10px;">{decision_text}</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi4:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #64748B; font-size: 0.85rem; font-weight: 600;">INFERENCE SPEED</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #3B82F6; margin-top: 8px;">{res.get('latency_ms', 0):.1f} ms</div>
                <div style="font-size: 0.75rem; color: #94A3B8;">{engine_source}</div>
            </div>
            """, unsafe_allow_html=True)

        # Progress bar
        st.markdown("<br>", unsafe_allow_html=True)
        st.write(f"**Churn Risk Score:** {prob_pct}%")
        st.progress(float(prob))

        # Retention Recommendations
        st.markdown("#### 🛡️ Actionable Customer Retention Strategies")
        recs = res.get("retention_recommendations", [])
        for r in recs:
            st.markdown(f'<div class="rec-box">💡 {r}</div>', unsafe_allow_html=True)

        # Risk Factors Evaluation
        with st.expander("🔍 Risk Drivers Deep Dive"):
            st.write("Key indicators influencing this customer's score:")
            indicators = []
            if "month-to-month" in contract.lower():
                indicators.append("⚠️ **Month-to-month contract**: High churn correlation observed in EDA.")
            if "electronic check" in payment_method.lower():
                indicators.append("⚠️ **Electronic check**: Highest churn rate payment channel.")
            if internet_service == "Fiber optic":
                indicators.append("⚠️ **Fiber optic service**: Higher service turnover observed.")
            if tenure <= 12:
                indicators.append("⚠️ **Early tenure tenure (<= 12m)**: Highest hazard rate period.")
            if tech_support == "Yes":
                indicators.append("✅ **Tech support active**: Acts as strong retention anchor.")
            if "two year" in contract.lower():
                indicators.append("✅ **Two-year contract**: Exceptional customer loyalty stabilizer.")

            for ind in indicators:
                st.write(f"- {ind}")

        with st.expander("📦 Raw API Response Payload"):
            st.json(res)


# =========================================================================
# TAB 2: BATCH PREDICTIONS
# =========================================================================
with tab2:
    st.subheader("Batch Customer Risk Scoring")
    st.write("Upload a CSV file containing customer records to receive batch predictions, probability scores, and retention tags.")

    uploaded_file = st.file_uploader("Upload Telecom Customers CSV", type=["csv"])

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        load_sample_batch = st.button("Load Cleaned Sample Batch (25 records)")

    batch_df = None
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
    elif load_sample_batch:
        sample_path = get_project_root() / "data" / "processed" / "cleaned_churn_data.csv"
        if sample_path.exists():
            batch_df = pd.read_csv(sample_path).head(25)
            st.success("Loaded 25 customer records from processed dataset.")

    if batch_df is not None:
        st.write(f"Loaded **{len(batch_df)}** records for evaluation.")
        st.dataframe(batch_df.head(5), use_container_width=True)

        if st.button("⚡ Score All Customers", type="primary"):
            predictor = get_predictor()
            with st.spinner(f"Scoring {len(batch_df)} customer records..."):
                t0 = time.time()
                results = []
                for _, row in batch_df.iterrows():
                    rec = row.to_dict()
                    pred_res = predictor.predict(rec)
                    results.append({
                        "customerID": rec.get("customerID", "Unknown"),
                        "Churn_Prediction": pred_res["churn_label"],
                        "Risk_Score_%": pred_res["risk_score_percent"],
                        "Risk_Level": pred_res["risk_level"],
                        "MonthlyCharges": rec.get("MonthlyCharges", 0),
                        "tenure": rec.get("tenure", 0),
                        "Contract": rec.get("Contract", "Unknown"),
                        "Primary_Recommendation": pred_res["retention_recommendations"][0] if pred_res["retention_recommendations"] else "Standard Support"
                    })
                total_duration = time.time() - t0

            scored_df = pd.DataFrame(results)
            st.success(f"Successfully scored {len(scored_df)} customers in {total_duration:.2f} seconds ({total_duration*1000/len(scored_df):.1f} ms/customer)!")

            # Risk Breakdown summary
            b1, b2, b3 = st.columns(3)
            high_count = (scored_df["Risk_Level"] == "High").sum()
            med_count = (scored_df["Risk_Level"] == "Medium").sum()
            low_count = (scored_df["Risk_Level"] == "Low").sum()
            b1.metric("High Risk Customers", f"{high_count} ({high_count*100/len(scored_df):.1f}%)")
            b2.metric("Medium Risk Customers", f"{med_count} ({med_count*100/len(scored_df):.1f}%)")
            b3.metric("Low Risk Customers", f"{low_count} ({low_count*100/len(scored_df):.1f}%)")

            # Scored Dataframe
            st.dataframe(scored_df, use_container_width=True)

            # Download CSV
            csv_data = scored_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Scored Customers CSV",
                data=csv_data,
                file_name="telecom_scored_customers.csv",
                mime="text/csv"
            )


# =========================================================================
# TAB 3: MODEL PERFORMANCE & ARCHITECTURE
# =========================================================================
with tab3:
    st.subheader("Production Machine Learning Architecture & Benchmark Metrics")

    meta_path = get_project_root() / "models" / "model_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
        m1, m2, m3, m4, m5 = st.columns(5)
        metrics = meta.get("test_set_metrics", {})
        m1.metric("Recall (Catch Churn)", f"{metrics.get('Recall', 0)*100:.2f}%", help="Primary retention metric")
        m2.metric("ROC-AUC Score", f"{metrics.get('ROC_AUC', 0):.4f}", help="Separation metric")
        m3.metric("F1-Score", f"{metrics.get('F1_Score', 0):.4f}", help="Harmonic mean of precision & recall")
        m4.metric("Accuracy", f"{metrics.get('Accuracy', 0)*100:.2f}%", help="Overall accuracy")
        m5.metric("Precision", f"{metrics.get('Precision', 0)*100:.2f}%", help="Budget efficiency")

        st.markdown("---")
        st.subheader("Model Configuration & Hyperparameters")
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Model Name:** `{meta.get('model_name')}`")
            st.write(f"**Version:** `{meta.get('model_version')}`")
            st.write(f"**Algorithm:** `{meta.get('algorithm')}`")
            st.write(f"**Features Evaluated:** `{meta.get('feature_count')}`")
        with c2:
            st.write("**Optimal Tuned Hyperparameters:**")
            st.json(meta.get("best_hyperparameters", {}))

    # Feature Importance table
    fi_path = get_project_root() / "reports" / "feature_importance_ranking.csv"
    if fi_path.exists():
        st.markdown("---")
        st.subheader("Top Predictive Features (Random Forest & Mutual Information)")
        fi_df = pd.read_csv(fi_path).head(15)
        st.dataframe(fi_df, use_container_width=True)
