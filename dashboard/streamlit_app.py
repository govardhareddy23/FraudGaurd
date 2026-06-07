import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="FraudGuard Portal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Endpoint definition
API_URL = st.sidebar.text_input("Backend API URL", "http://localhost:8000/api")

# Custom styling for professional dark mode theme
st.markdown("""
    <style>
    .main {
        background-color: #0b111e;
        color: #c9d1d9;
    }
    .stButton>button {
        background-color: #00e5b4;
        color: #080f18;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #00b890;
        color: #080f18;
        transform: translateY(-2px);
    }
    .metric-card {
        background-color: #0d1527;
        border: 1px solid #1f2e4d;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


def get_kpis():
    try:
        r = requests.get(f"{API_URL}/dashboard/stats")
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    # Fallback default values
    return {
        "total_transactions": 0, "fraud_transactions": 0, "fraud_rate": 0.0,
        "avg_amount": 0.0, "avg_fraud_amount": 0.0, "open_alerts": 0,
        "transactions_today": 0, "fraud_today": 0
    }


def list_alerts():
    try:
        r = requests.get(f"{API_URL}/alerts/?status=open")
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


# Sidebar Header
st.sidebar.image("https://img.icons8.com/nolan/128/security-shield.png", width=80)
st.sidebar.title("FraudGuard AI")
st.sidebar.markdown("Deep learning Credit Card Fraud Protection center.")

# Navigation
page = st.sidebar.radio("Navigation Menu", ["Real-Time Dashboard", "New Transaction Entry", "Fraud Alerts Queue"])

# ── PAGE 1: Real-Time Dashboard ──
if page == "Real-Time Dashboard":
    st.title("🛡️ Real-Time Monitoring Portal")
    st.write("Live status check of transactions feed processed by deep ANN.")
    
    kpis = get_kpis()

    # Column layout for metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><p style="color:#8b949e;font-size:12px;">TOTAL TRANSACTIONS</p><h2 style="color:#ffffff;">{kpis["total_transactions"]:,}</h2></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><p style="color:#8b949e;font-size:12px;">FRAUD TRANSACTIONS</p><h2 style="color:#ff4444;">{kpis["fraud_transactions"]:,}</h2></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><p style="color:#8b949e;font-size:12px;">FRAUD RATE</p><h2 style="color:#ffaa00;">{kpis["fraud_rate"]:.4f}%</h2></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><p style="color:#8b949e;font-size:12px;">PENDING ALERTS</p><h2 style="color:#ff4444;animation: pulse 2s infinite;">{kpis["open_alerts"]}</h2></div>', unsafe_allow_html=True)

    st.write("")

    # Visual graphs section
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Transaction Log History")
        try:
            r = requests.get(f"{API_URL}/transactions/?limit=15")
            if r.status_code == 200:
                tx_list = r.json()
                if tx_list:
                    df_tx = pd.DataFrame(tx_list)
                    # Format output columns
                    df_tx["timestamp"] = pd.to_datetime(df_tx["timestamp"])
                    df_tx["amount"] = df_tx["amount"].apply(lambda x: f"${x:.2f}")
                    df_tx["fraud_probability"] = df_tx["fraud_probability"].apply(lambda x: f"{x*100:.2f}%")
                    st.dataframe(df_tx[["id", "timestamp", "amount", "fraud_probability", "risk_level", "is_fraud"]], use_container_width=True)
                else:
                    st.info("No transaction records in DB. Submit a prediction first.")
        except Exception as e:
            st.error(f"Cannot load transactions: {e}")

    with c2:
        st.subheader("Hourly Fraud Rate Distribution")
        try:
            r = requests.get(f"{API_URL}/dashboard/chart/hourly")
            if r.status_code == 200 and r.json():
                chart_data = pd.DataFrame(r.json())
                fig = px.bar(chart_data, x="hour", y="fraud", title="Fraud Count by Hour of Day", color_discrete_sequence=["#ff4444"])
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Mock graph if DB is blank
                mock_data = pd.DataFrame({"hour": list(range(24)), "fraud": np.random.randint(0, 5, 24)})
                fig = px.line(mock_data, x="hour", y="fraud", title="Fraud Trend by Hour (Example Data)", markers=True)
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Start prediction inputs to feed the chart.")

# ── PAGE 2: New Transaction Entry ──
elif page == "New Transaction Entry":
    st.title("💳 Transaction Scoring Console")
    st.write("Simulate or paste credit card attributes to evaluate risk.")

    # Presets selectbox
    preset = st.selectbox("Quick Transaction Profile Presets", [
        "Normal Legitimate Purchase ($149.62)",
        "Known Outlier Fraud Pattern ($9.25)",
        "Suspicious Transaction ($45.00)"
    ])

    # Preset values map
    preset_vals = {
        "Normal Legitimate Purchase ($149.62)": {"amount": 149.62, "time": 400.0, "v14": 0.31, "v17": 0.58},
        "Known Outlier Fraud Pattern ($9.25)": {"amount": 9.25, "time": 1000.0, "v14": -7.03, "v17": -6.24},
        "Suspicious Transaction ($45.00)": {"amount": 45.00, "time": 850.0, "v14": -2.5, "v17": -1.8}
    }

    # Gather inputs
    p_data = preset_vals[preset]
    
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=p_data["amount"])
        time_val = st.number_input("Transaction Seconds offset (Time)", min_value=0.0, value=p_data["time"])
    with col2:
        v14 = st.number_input("V14 (High Fraud Correlation Feature)", value=p_data["v14"])
        v17 = st.number_input("V17 (High Fraud Correlation Feature)", value=p_data["v17"])

    # Expandable advanced V1-V28 inputs
    with st.expander("Show Advanced PCA Features (V1-V28)"):
        pca_dict = {}
        adv_cols = st.columns(4)
        for i in range(1, 29):
            col_idx = (i - 1) % 4
            default_val = v14 if i == 14 else (v17 if i == 17 else 0.0)
            pca_dict[f"v{i}"] = adv_cols[col_idx].number_input(f"v{i}", value=default_val, key=f"v_field_{i}")

    if st.button("Run Artificial Neural Network Evaluation"):
        # Compile request
        payload = {
            "time": time_val,
            "amount": amount,
            **pca_dict
        }

        with st.spinner("ANN calculating fraud risk..."):
            try:
                r = requests.post(f"{API_URL}/predict/", json=payload)
                if r.status_code == 200:
                    res = r.json()
                    prob = res["fraud_probability"]
                    
                    # Risk results display
                    st.write("---")
                    res_col1, res_col2 = st.columns([1, 2])
                    
                    with res_col1:
                        # Gauge plot for probability visualization
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prob * 100,
                            title={'text': "Fraud Probability (%)"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "#ff4444" if prob >= 0.50 else "#00e5b4"},
                                'steps': [
                                    {'range': [0, 40], 'color': "rgba(0, 229, 180, 0.2)"},
                                    {'range': [40, 80], 'color': "rgba(255, 170, 0, 0.2)"},
                                    {'range': [80, 100], 'color': "rgba(255, 68, 68, 0.2)"}
                                ],
                            }
                        ))
                        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9", height=250)
                        st.plotly_chart(fig, use_container_width=True)

                    with res_col2:
                        st.subheader("Decision Breakdown")
                        color = "#ff4444" if res["is_fraud"] else "#00e5b4"
                        st.markdown(f"Status: <b style='color:{color};font-size:22px;'>{'FRAUD DETECTED' if res['is_fraud'] else 'TRANSACTION APPROVED'}</b>", unsafe_allow_html=True)
                        st.markdown(f"Risk Level: **{res['risk_level']}**")
                        st.markdown(f"Alert Status: **{'TRIGGERED (>80%)' if res['alert_triggered'] else 'None'}**")
                        st.markdown(f"Unique Database ID: **#{res['transaction_id']}**")

                    # Explainable AI call
                    st.subheader("🔮 SHAP Feature Importance Explainer")
                    with st.spinner("Computing Shapley additive values..."):
                        try:
                            r_exp = requests.post(f"{API_URL}/predict/{res['transaction_id']}/explain")
                            if r_exp.status_code == 200:
                                exp = r_exp.json()
                                contribs = exp["contributions"]
                                
                                # Make dataframe for plotting
                                df_exp = pd.DataFrame([{"Feature": k, "SHAP Value": v} for k, v in contribs.items() if abs(v) > 0.001])
                                if not df_exp.empty:
                                    df_exp = df_exp.sort_values(by="SHAP Value", key=abs, ascending=True).tail(10)
                                    fig_exp = px.bar(df_exp, x="SHAP Value", y="Feature", orientation="h",
                                                     title="Top Features Influencing Prediction Decision",
                                                     color="SHAP Value",
                                                     color_continuous_scale="RdYlGn_r")
                                    fig_exp.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9")
                                    st.plotly_chart(fig_exp, use_container_width=True)
                                else:
                                    st.info("SHAP values negligible for this safe prediction.")
                        except Exception as e:
                            st.warning(f"Could not compute SHAP graph: {e}")
                else:
                    st.error("Backend returned prediction error. Make sure FastAPI server is running.")
            except Exception as e:
                st.error(f"Cannot contact API server: {e}")

# ── PAGE 3: Fraud Alerts Queue ──
elif page == "Fraud Alerts Queue":
    st.title("🚨 High-Risk Alerts Review")
    st.write("Review queue containing flagged transactions (Risk > 80%). Verify actions below.")
    
    alerts = list_alerts()
    
    if not alerts:
        st.success("Clear Queue! No open high-risk fraud alerts pending review.")
    else:
        for a in alerts:
            with st.container():
                st.markdown(f"""
                    <div style="background-color:#210e14; border-left:4px solid #ff4444; padding:15px; border-radius:6px; margin-bottom:12px;">
                        <span style="color:#ff6666; font-size:11px; font-weight:bold;">ALERT ID #{a['id']}</span>
                        <h4 style="color:#ff4444; margin:2px 0;">{a['fraud_probability']*100:.1f}% Fraud Probability Detected</h4>
                        <p style="margin:2px 0; color:#c9d1d9;">Amount Flagged: <b>${a['amount']:.2f}</b> | Transaction Source ID: #{a['transaction_id']}</p>
                        <p style="margin:2px 0; font-size:11px; color:#8b949e;">Flagged Time: {a['timestamp']}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Review actions
                ac1, ac2, ac3 = st.columns([1, 1, 6])
                if ac1.button("Reviewed", key=f"rev_{a['id']}"):
                    try:
                        requests.patch(f"{API_URL}/alerts/{a['id']}?status=reviewed")
                        st.rerun()
                    except Exception:
                        pass
                if ac2.button("Dismiss", key=f"dism_{a['id']}"):
                    try:
                        requests.patch(f"{API_URL}/alerts/{a['id']}?status=dismissed")
                        st.rerun()
                    except Exception:
                        pass
                st.write("")
