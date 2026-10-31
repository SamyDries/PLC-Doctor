
from pathlib import Path
from datetime import datetime
import uuid

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from gemini_assistant import generate_diagnosis


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "plc_fault_model.joblib"

st.set_page_config(
    page_title="PLC Doctor | Maintenance Management",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. DESIGN
# =========================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background-color: #f5f7fb;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #263244;
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    .brand {
        font-size: 25px;
        font-weight: 700;
        color: white;
        padding: 8px 0 2px 0;
    }

    .brand-subtitle {
        color: #9ca3af;
        font-size: 12px;
        margin-bottom: 28px;
    }

    .page-title {
        font-size: 29px;
        font-weight: 700;
        color: #172033;
        margin-bottom: 3px;
    }

    .page-subtitle {
        color: #718096;
        font-size: 14px;
        margin-bottom: 25px;
    }

    .metric-card {
        background: white;
        padding: 20px;
        border: 1px solid #e7ebf2;
        border-radius: 12px;
        min-height: 112px;
    }

    .metric-label {
        color: #718096;
        font-size: 13px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #172033;
        font-size: 27px;
        font-weight: 700;
    }

    .panel {
        background: white;
        padding: 22px;
        border: 1px solid #e7ebf2;
        border-radius: 12px;
        margin-bottom: 16px;
    }

    .panel-title {
        font-size: 17px;
        font-weight: 700;
        color: #172033;
        margin-bottom: 12px;
    }

    .status-open {
        color: #b45309;
        font-weight: 600;
    }

    .status-progress {
        color: #2563eb;
        font-weight: 600;
    }

    .status-completed {
        color: #15803d;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"] {
        background: #2563eb;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        min-height: 42px;
    }

    div.stButton > button {
        border-radius: 8px;
        min-height: 38px;
    }

    div[data-testid="stForm"] {
        background: white;
        border: 1px solid #e7ebf2;
        border-radius: 12px;
        padding: 20px;
    }

    .notice {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 13px 16px;
        border-radius: 5px;
        color: #1e3a8a;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOAD APPLICATION RESOURCES
# =========================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


model = load_model()


# =========================================================
# 4. SESSION DATA
# =========================================================

def initialize_state():

    if "assets" not in st.session_state:
        st.session_state.assets = [
            {
                "Asset ID": "MOT-001",
                "Equipment": "Electric Motor",
                "Location": "Production Line 1",
                "Manufacturer": "Demo Manufacturer",
                "Status": "Operational"
            },
            {
                "Asset ID": "VFD-001",
                "Equipment": "Variable Frequency Drive",
                "Location": "Production Line 1",
                "Manufacturer": "Demo Manufacturer",
                "Status": "Operational"
            },
            {
                "Asset ID": "PLC-001",
                "Equipment": "PLC Controller",
                "Location": "Control Cabinet",
                "Manufacturer": "Demo Manufacturer",
                "Status": "Operational"
            }
        ]

    if "work_orders" not in st.session_state:
        st.session_state.work_orders = []

    if "history" not in st.session_state:
        st.session_state.history = []

    if "last_diagnosis" not in st.session_state:
        st.session_state.last_diagnosis = None


initialize_state()


# =========================================================
# 5. COMMON FUNCTIONS
# =========================================================

def page_header(title, subtitle):
    st.markdown(f"""
        <div class="page-title">{title}</div>
        <div class="page-subtitle">{subtitle}</div>
    """, unsafe_allow_html=True)


def metric_card(label, value):
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
    """, unsafe_allow_html=True)


def get_asset(asset_id):
    for asset in st.session_state.assets:
        if asset["Asset ID"] == asset_id:
            return asset
    return None


def create_diagnosis(features, asset_id):

    if model is None:
        st.error(
            "The diagnostic service is unavailable. "
            "Please check the application configuration."
        )
        return

    input_data = pd.DataFrame([features])

    prediction = model.predict(input_data)[0]

    asset = get_asset(asset_id)
    asset_name = asset["Equipment"] if asset else "Unknown equipment"

    result = {
        "id": str(uuid.uuid4())[:8].upper(),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "asset_id": asset_id,
        "asset_name": asset_name,
        "fault": prediction,
        "features": features
    }

    st.session_state.last_diagnosis = result
    st.session_state.history.insert(0, result)

    return result


def priority_for_fault(fault):

    if fault in ["Motor Overload", "Power Supply Failure"]:
        return "High"

    if fault in ["VFD Fault", "PLC Communication Failure"]:
        return "High"

    if fault == "Sensor Failure":
        return "Medium"

    return "Low"


# =========================================================
# 6. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown('<div class="brand">⚙️ PLC Doctor</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="brand-subtitle">INDUSTRIAL MAINTENANCE PLATFORM</div>',
        unsafe_allow_html=True
    )

    st.markdown("**WORKSPACE**")

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Asset Management",
            "New Diagnosis",
            "Work Orders",
            "Maintenance Records"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("**SYSTEM STATUS**")

    if model is not None:
        st.success("Diagnostic service available")
    else:
        st.error("Diagnostic service unavailable")

    st.markdown("---")
    st.caption("PLC Doctor · Industrial Maintenance")
    st.caption("Prototype version")


# =========================================================
# 7. OVERVIEW
# =========================================================

if page == "Overview":

    page_header(
        "Maintenance Overview",
        "Monitor your equipment, diagnostics and maintenance activities."
    )

    total_assets = len(st.session_state.assets)
    total_orders = len(st.session_state.work_orders)
    open_orders = sum(
        1 for order in st.session_state.work_orders
        if order["Status"] == "Open"
    )
    completed_orders = sum(
        1 for order in st.session_state.work_orders
        if order["Status"] == "Completed"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Registered assets", total_assets)

    with c2:
        metric_card("Total work orders", total_orders)

    with c3:
        metric_card("Open work orders", open_orders)

    with c4:
        metric_card("Completed orders", completed_orders)

    st.markdown("")

    left, right = st.columns([1.5, 1])

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Recent diagnostics</div>',
                    unsafe_allow_html=True)

        if st.session_state.history:
            recent = []

            for item in st.session_state.history[:5]:
                recent.append({
                    "Date": item["date"],
                    "Asset": item["asset_id"],
                    "Equipment": item["asset_name"],
                    "Result": item["fault"]
                })

            st.dataframe(
                pd.DataFrame(recent),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No diagnostics have been recorded yet.")

        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Work order status</div>',
                    unsafe_allow_html=True)

        if st.session_state.work_orders:
            status_counts = (
                pd.DataFrame(st.session_state.work_orders)["Status"]
                .value_counts()
                .reset_index()
            )
            status_counts.columns = ["Status", "Count"]

            fig = px.pie(
                status_counts,
                names="Status",
                values="Count",
                hole=0.62
            )
            fig.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                showlegend=True,
                height=250
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Work order statistics will appear here.")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Quick actions")

    q1, q2, q3 = st.columns(3)

    with q1:
        if st.button("＋ Register equipment", use_container_width=True):
            st.session_state.quick_navigation = "Asset Management"
            st.rerun()

    with q2:
        if st.button("＋ Start diagnosis", use_container_width=True):
            st.session_state.quick_navigation = "New Diagnosis"
            st.rerun()

    with q3:
        if st.button("＋ View work orders", use_container_width=True):
            st.session_state.quick_navigation = "Work Orders"
            st.rerun()


# =========================================================
# 8. ASSET MANAGEMENT
# =========================================================

elif page == "Asset Management":

    page_header(
        "Asset Management",
        "Register and manage the industrial equipment in your facility."
    )

    st.markdown("### Equipment register")

    assets_df = pd.DataFrame(st.session_state.assets)

    if not assets_df.empty:
        st.dataframe(
            assets_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No equipment registered yet.")

    st.markdown("### Register new equipment")

    with st.form("add_asset_form", clear_on_submit=True):

        col1, col2 = st.columns(2)

        with col1:
            asset_id = st.text_input(
                "Asset ID",
                placeholder="e.g. MOT-002"
            )

            equipment = st.selectbox(
                "Equipment type",
                [
                    "Electric Motor",
                    "Variable Frequency Drive",
                    "PLC Controller",
                    "Industrial Sensor",
                    "Pump",
                    "Conveyor",
                    "Other"
                ]
            )

        with col2:
            location = st.text_input(
                "Location",
                placeholder="e.g. Production Line 2"
            )

            manufacturer = st.text_input(
                "Manufacturer",
                placeholder="e.g. Siemens"
            )

        submitted = st.form_submit_button(
            "Register equipment",
            type="primary",
            use_container_width=True
        )

        if submitted:

            if not asset_id.strip() or not location.strip():
                st.error("Please enter an Asset ID and a location.")

            elif any(
                a["Asset ID"].lower() == asset_id.strip().lower()
                for a in st.session_state.assets
            ):
                st.error("This Asset ID already exists.")

            else:
                st.session_state.assets.append({
                    "Asset ID": asset_id.strip().upper(),
                    "Equipment": equipment,
                    "Location": location.strip(),
                    "Manufacturer": manufacturer.strip() or "Not specified",
                    "Status": "Operational"
                })

                st.success("Equipment registered successfully.")
                st.rerun()


# =========================================================
# 9. NEW DIAGNOSIS
# =========================================================

elif page == "New Diagnosis":

    page_header(
        "Equipment Diagnosis",
        "Enter machine measurements to investigate an equipment problem."
    )

    if not st.session_state.assets:
        st.warning("Register equipment before starting a diagnosis.")

    else:

        asset_options = {
            f'{a["Asset ID"]} — {a["Equipment"]}': a["Asset ID"]
            for a in st.session_state.assets
        }

        with st.form("diagnosis_form"):

            st.markdown("#### 1. Select equipment")

            selected_asset_label = st.selectbox(
                "Equipment",
                list(asset_options.keys())
            )

            selected_asset_id = asset_options[selected_asset_label]
            selected_asset = get_asset(selected_asset_id)

            st.caption(
                f'Location: {selected_asset["Location"]} '
                f'| Status: {selected_asset["Status"]}'
            )

            st.markdown("---")
            st.markdown("#### 2. Motor measurements")

            c1, c2, c3 = st.columns(3)

            with c1:
                motor_current = st.number_input(
                    "Motor current (A)",
                    min_value=0.0,
                    max_value=100.0,
                    value=8.0,
                    step=0.5
                )

            with c2:
                motor_temperature = st.number_input(
                    "Motor temperature (°C)",
                    min_value=0.0,
                    max_value=200.0,
                    value=45.0,
                    step=1.0
                )

            with c3:
                vibration = st.number_input(
                    "Vibration (mm/s)",
                    min_value=0.0,
                    max_value=50.0,
                    value=1.5,
                    step=0.1
                )

            st.markdown("---")
            st.markdown("#### 3. Electrical measurements")

            c1, c2 = st.columns(2)

            with c1:
                voltage = st.number_input(
                    "Supply voltage (V)",
                    min_value=0.0,
                    max_value=500.0,
                    value=230.0,
                    step=1.0
                )

            with c2:
                machine_running = st.selectbox(
                    "Machine operating status",
                    ["Running", "Stopped"]
                )

            st.markdown("---")
            st.markdown("#### 4. Automation and control")

            c1, c2, c3 = st.columns(3)

            with c1:
                sensor_status = st.selectbox(
                    "Sensor status",
                    ["Operational", "Failure detected"]
                )

            with c2:
                vfd_fault = st.selectbox(
                    "VFD fault indicator",
                    ["No fault", "Fault detected"]
                )

            with c3:
                plc_communication = st.selectbox(
                    "PLC communication",
                    ["Connected", "Communication failure"]
                )

            st.markdown("---")

            submitted = st.form_submit_button(
                "Run equipment diagnosis",
                type="primary",
                use_container_width=True
            )

            if submitted:

                features = {
                    "motor_current": motor_current,
                    "motor_temperature": motor_temperature,
                    "vibration": vibration,
                    "sensor_status": (
                        1 if sensor_status == "Operational" else 0
                    ),
                    "voltage": voltage,
                    "vfd_fault": (
                        1 if vfd_fault == "Fault detected" else 0
                    ),
                    "plc_communication": (
                        1 if plc_communication == "Connected" else 0
                    ),
                    "machine_running": (
                        1 if machine_running == "Running" else 0
                    )
                }

                result = create_diagnosis(
                    features,
                    selected_asset_id
                )

                if result:
                    st.success("Diagnosis completed.")
                    st.rerun()

    # Display latest diagnosis
    if st.session_state.last_diagnosis:

        result = st.session_state.last_diagnosis

        st.markdown("---")
        st.markdown("### Latest diagnostic result")

        r1, r2 = st.columns([1, 2])

        with r1:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.caption("EQUIPMENT")
            st.subheader(result["asset_id"])
            st.write(result["asset_name"])
            st.caption(result["date"])
            st.markdown('</div>', unsafe_allow_html=True)

        with r2:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.caption("IDENTIFIED CONDITION")
            st.subheader(result["fault"])

            if result["fault"] == "Normal Operation":
                st.success("No abnormal condition identified.")
            else:
                st.warning(
                    "A potential equipment issue was identified. "
                    "Review the technical report and verify the measurements."
                )

            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### AI technical report")

        if st.button("Generate technical report", type="primary"):

            with st.spinner("Preparing technical analysis..."):

                report = generate_diagnosis(
                    fault_type=result["fault"],
                    probability=0,
                    machine_data=result["features"],
                    language="English"
                )

                st.session_state.last_report = report

        if "last_report" in st.session_state:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown(st.session_state.last_report)
            st.markdown('</div>', unsafe_allow_html=True)

        if result["fault"] != "Normal Operation":

            if st.button("Create work order from this diagnosis"):
                st.session_state.prefill_work_order = True
                st.rerun()


# =========================================================
# 10. WORK ORDERS
# =========================================================

elif page == "Work Orders":

    page_header(
        "Maintenance Work Orders",
        "Create, assign and track maintenance interventions."
    )

    orders = st.session_state.work_orders

    total = len(orders)
    open_count = sum(1 for o in orders if o["Status"] == "Open")
    progress_count = sum(
        1 for o in orders if o["Status"] == "In Progress"
    )
    completed_count = sum(
        1 for o in orders if o["Status"] == "Completed"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Total orders", total)

    with c2:
        metric_card("Open", open_count)

    with c3:
        metric_card("In progress", progress_count)

    with c4:
        metric_card("Completed", completed_count)

    st.markdown("### Create a work order")

    last = st.session_state.last_diagnosis
    prefill = st.session_state.get("prefill_work_order", False)

    with st.form("work_order_form", clear_on_submit=True):

        if last and prefill:
            default_asset = last["asset_id"]
            default_fault = last["fault"]
        else:
            default_asset = (
                st.session_state.assets[0]["Asset ID"]
                if st.session_state.assets else ""
            )
            default_fault = ""

        asset_ids = [a["Asset ID"] for a in st.session_state.assets]

        if asset_ids:
            asset_index = (
                asset_ids.index(default_asset)
                if default_asset in asset_ids else 0
            )

            asset_id = st.selectbox(
                "Equipment",
                asset_ids,
                index=asset_index
            )
        else:
            asset_id = ""

        issue = st.text_input(
            "Issue / maintenance task",
            value=(
                f"Investigate: {default_fault}"
                if prefill and default_fault else ""
            ),
            placeholder="Describe the maintenance task"
        )

        c1, c2 = st.columns(2)

        with c1:
            priority = st.selectbox(
                "Priority",
                ["Low", "Medium", "High", "Critical"],
                index=(
                    2 if prefill and default_fault
                    and default_fault != "Sensor Failure" else 1
                )
            )

        with c2:
            technician = st.text_input(
                "Assigned technician",
                placeholder="Technician name"
            )

        description = st.text_area(
            "Work description",
            placeholder="Describe the inspection or repair to be performed..."
        )

        submitted = st.form_submit_button(
            "Create work order",
            type="primary",
            use_container_width=True
        )

        if submitted:

            if not asset_id or not issue.strip():
                st.error("Select equipment and enter a maintenance task.")

            else:

                order = {
                    "Order ID": f"WO-{len(orders) + 1:04d}",
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Asset ID": asset_id,
                    "Issue": issue.strip(),
                    "Priority": priority,
                    "Technician": technician.strip() or "Unassigned",
                    "Description": description.strip(),
                    "Status": "Open"
                }

                st.session_state.work_orders.insert(0, order)
                st.session_state.prefill_work_order = False

                st.success(f'Work order {order["Order ID"]} created.')
                st.rerun()

    st.markdown("---")
    st.markdown("### Work order register")

    if orders:

        for index, order in enumerate(orders):

            with st.expander(
                f'{order["Order ID"]}  |  {order["Asset ID"]}  |  '
                f'{order["Issue"]}'
            ):

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.caption("PRIORITY")
                    st.write(order["Priority"])

                with c2:
                    st.caption("ASSIGNED TO")
                    st.write(order["Technician"])

                with c3:
                    st.caption("CREATED")
                    st.write(order["Date"])

                st.write("**Description**")
                st.write(order["Description"] or "No additional details.")

                st.write("**Status**")

                new_status = st.selectbox(
                    "Update status",
                    ["Open", "In Progress", "Completed"],
                    index=[
                        "Open", "In Progress", "Completed"
                    ].index(order["Status"]),
                    key=f"status_{order['Order ID']}"
                )

                if new_status != order["Status"]:
                    st.session_state.work_orders[index]["Status"] = new_status
                    st.rerun()

    else:
        st.info("No work orders yet. Create one using the form above.")


# =========================================================
# 11. MAINTENANCE RECORDS
# =========================================================

elif page == "Maintenance Records":

    page_header(
        "Maintenance Records",
        "Review diagnostic history and maintenance interventions."
    )

    st.markdown("### Diagnostic history")

    if st.session_state.history:

        history_data = []

        for item in st.session_state.history:
            history_data.append({
                "Date": item["date"],
                "Diagnostic ID": item["id"],
                "Asset ID": item["asset_id"],
                "Equipment": item["asset_name"],
                "Identified condition": item["fault"]
            })

        history_df = pd.DataFrame(history_data)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download diagnostic history (CSV)",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="plc_doctor_diagnostic_history.csv",
            mime="text/csv"
        )

    else:
        st.info("No diagnostic records available.")

    st.markdown("---")
    st.markdown("### Work order history")

    if st.session_state.work_orders:

        orders_df = pd.DataFrame(st.session_state.work_orders)

        st.dataframe(
            orders_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download work orders (CSV)",
            data=orders_df.to_csv(index=False).encode("utf-8"),
            file_name="plc_doctor_work_orders.csv",
            mime="text/csv"
        )

    else:
        st.info("No maintenance interventions recorded yet.")

    st.markdown("---")
    st.markdown("### Equipment register")

    if st.session_state.assets:

        st.dataframe(
            pd.DataFrame(st.session_state.assets),
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# 12. SAFETY NOTICE
# =========================================================

st.markdown("---")

st.markdown("""
<div class="notice">
    <strong>Maintenance safety notice:</strong>
    Diagnostic results are decision-support information.
    Always verify equipment conditions through physical inspection
    and follow the manufacturer's procedures and industrial safety rules.
    This prototype does not replace qualified maintenance personnel.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; padding:20px; color:#64748b;'>
    <strong style='color:#172033;'>PLC Doctor</strong>
    <br>
    Industrial Maintenance Platform
    <br>
    <span style='font-size:13px;'>Developed by Samy Dries © 2026</span>
</div>
""", unsafe_allow_html=True)
