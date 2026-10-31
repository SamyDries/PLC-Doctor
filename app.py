
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from gemini_assistant import generate_diagnosis


# -----------------------------
# PAGE CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="PLC Doctor",
    page_icon="⚙️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "plc_fault_model.joblib"

FEATURES = [
    "motor_current",
    "motor_temperature",
    "vibration",
    "sensor_status",
    "voltage",
    "vfd_fault",
    "plc_communication",
    "machine_running"
]

# -----------------------------
# LOAD MODEL
# -----------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

if "history" not in st.session_state:
    st.session_state.history = []


# -----------------------------
# CUSTOM STYLE
# -----------------------------

st.markdown("""
<style>
.stApp {
    background-color: #0b1220;
    color: #f1f5f9;
}
.block-container {
    padding-top: 2rem;
}
div[data-testid="stMetric"] {
    background: #172337;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #263750;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# HEADER
# -----------------------------

st.title("⚙️ PLC Doctor")
st.markdown(
    "### Industrial Fault Diagnosis Assistant"
)
st.caption(
    "Machine Learning + Gemini AI | "
    "Preliminary industrial maintenance diagnosis"
)

st.divider()

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("PLC Doctor")
st.sidebar.caption("Industrial diagnostic platform")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Fault Diagnosis", "Diagnostic History"]
)

st.sidebar.divider()
st.sidebar.info(
    "Educational prototype using a synthetic dataset. "
    "Not a certified industrial safety or diagnostic system."
)


# -----------------------------
# DASHBOARD
# -----------------------------

if page == "Dashboard":

    st.subheader("System Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("ML Model", "Decision Tree")
    col2.metric("Fault Categories", "6")
    col3.metric("AI Assistant", "Gemini")

    st.divider()

    st.subheader("Supported Fault Categories")

    faults = [
        ("Motor Overload", "High current and motor temperature"),
        ("Sensor Failure", "Missing or faulty sensor signal"),
        ("Power Supply Failure", "Abnormal supply voltage"),
        ("VFD Fault", "Variable Frequency Drive fault"),
        ("PLC Communication Failure", "Communication interruption"),
        ("Normal Operation", "No simulated fault detected")
    ]

    cols = st.columns(3)

    for i, (name, description) in enumerate(faults):
        with cols[i % 3]:
            st.markdown(f"**{name}**")
            st.caption(description)

    st.divider()

    st.write(
        "Select **Fault Diagnosis** in the sidebar "
        "to analyze machine operating conditions."
    )


# -----------------------------
# FAULT DIAGNOSIS
# -----------------------------

elif page == "Fault Diagnosis":

    st.subheader("Machine Parameters")

    st.write(
        "Enter the machine measurements and operating states."
    )

    with st.form("diagnosis_form"):

        col1, col2 = st.columns(2)

        with col1:
            motor_current = st.number_input(
                "Motor Current (A)",
                min_value=0.0,
                max_value=100.0,
                value=25.5,
                step=0.5
            )

            motor_temperature = st.number_input(
                "Motor Temperature (°C)",
                min_value=0.0,
                max_value=200.0,
                value=95.0,
                step=1.0
            )

            vibration = st.number_input(
                "Vibration (mm/s)",
                min_value=0.0,
                max_value=50.0,
                value=4.2,
                step=0.1
            )

            voltage = st.number_input(
                "Supply Voltage (V)",
                min_value=0.0,
                max_value=500.0,
                value=230.0,
                step=1.0
            )

        with col2:
            sensor_status = st.selectbox(
                "Sensor Status",
                ["Operational", "Failure"]
            )

            vfd_fault = st.selectbox(
                "VFD Status",
                ["No Fault", "Fault Detected"]
            )

            plc_communication = st.selectbox(
                "PLC Communication",
                ["Connected", "Disconnected"]
            )

            machine_running = st.selectbox(
                "Machine Status",
                ["Running", "Stopped"]
            )

        submitted = st.form_submit_button(
            "Analyze Machine",
            type="primary",
            use_container_width=True
        )

    if submitted:

        machine_data = {
            "motor_current": motor_current,
            "motor_temperature": motor_temperature,
            "vibration": vibration,
            "sensor_status": 1 if sensor_status == "Operational" else 0,
            "voltage": voltage,
            "vfd_fault": 1 if vfd_fault == "Fault Detected" else 0,
            "plc_communication": (
                1 if plc_communication == "Connected" else 0
            ),
            "machine_running": (
                1 if machine_running == "Running" else 0
            )
        }

        input_df = pd.DataFrame(
            [machine_data],
            columns=FEATURES
        )

        prediction = model.predict(input_df)[0]

        probabilities = model.predict_proba(input_df)[0]

        probability = float(max(probabilities)) * 100

        st.session_state.last_diagnosis = {
            "fault": prediction,
            "probability": probability,
            "data": machine_data,
            "probabilities": probabilities
        }

        st.session_state.history.append({
            "Fault": prediction,
            "Confidence (%)": round(probability, 2)
        })

    # DISPLAY RESULTS

    if "last_diagnosis" in st.session_state:

        result = st.session_state.last_diagnosis

        st.divider()
        st.subheader("Machine Learning Prediction")

        col1, col2 = st.columns(2)

        col1.metric(
            "Predicted Condition",
            result["fault"]
        )

        col2.metric(
            "Model Confidence",
            f'{result["probability"]:.2f}%'
        )

        if result["fault"] == "Normal Operation":
            st.success("No simulated fault detected.")
        else:
            st.warning(
                "Potential fault detected. "
                "Physical verification is required."
            )

        probability_df = pd.DataFrame({
            "Fault Category": model.classes_,
            "Probability (%)": result["probabilities"] * 100
        }).sort_values("Probability (%)", ascending=True)

        fig = px.bar(
            probability_df,
            x="Probability (%)",
            y="Fault Category",
            orientation="h",
            title="Fault Classification Probabilities",
            range_x=[0, 100]
        )

        fig.update_layout(
            paper_bgcolor="#0b1220",
            plot_bgcolor="#172337",
            font_color="white"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Gemini AI Diagnostic Report")

        language = st.selectbox(
            "Report Language",
            ["English", "French"]
        )

        if st.button(
            "Generate AI Explanation",
            type="primary"
        ):

            with st.spinner("Gemini is analyzing the diagnosis..."):

                report = generate_diagnosis(
                    fault_type=result["fault"],
                    probability=result["probability"],
                    machine_data=result["data"],
                    language=language
                )

            st.session_state.last_report = report

        if "last_report" in st.session_state:
            st.markdown(st.session_state.last_report)

        st.caption(
            "AI output is advisory only. "
            "Always follow official safety procedures."
        )


# -----------------------------
# DIAGNOSTIC HISTORY
# -----------------------------

elif page == "Diagnostic History":

    st.subheader("Diagnostic History")

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True
        )

        if st.button("Clear History"):
            st.session_state.history = []
            st.session_state.pop("last_diagnosis", None)
            st.session_state.pop("last_report", None)
            st.rerun()

    else:
        st.info("No diagnostic analyses have been performed yet.")
