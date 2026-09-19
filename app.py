"""KrishiKavach Streamlit application."""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
from PIL import Image
from streamlit_js_eval import get_geolocation
from torchvision import transforms

from knowledge_base import CROP_KNOWLEDGE_BASE
from model import GradCAM, apply_heatmap_overlay, get_crop_model
from translations import LANGUAGES, get_translation

MODEL_PATH = "crop_disease_model.pth"

st.set_page_config(page_title="KrishiKavach", page_icon="🌾", layout="wide")

st.markdown(
    """<style>
    .main-header {
        color:#0f3d2e;
        text-align:center;
        font-weight:900;
        font-size:2.8rem;
        margin-bottom:0.2rem;
        letter-spacing:0.02em;
    }
    .sub-header {
        color:#2e7d32;
        text-align:center;
        font-weight:700;
        margin-bottom:1.5rem;
        font-size:1.15rem;
    }
    .hero-panel {
        background: linear-gradient(135deg, #e9f8ed 0%, #f4fff8 40%, #edf7ff 100%);
        border: 1px solid rgba(39, 125, 62, 0.15);
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 8px 24px rgba(40, 100, 60, 0.08);
        margin-bottom: 1.25rem;
    }
    .feature-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(239,247,240,0.9));
        border: 1px solid rgba(26, 86, 42, 0.12);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 5px 16px rgba(40,80,55,0.08);
        height: 100%;
    }
    .status-pill {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        background: rgba(30, 120, 60, 0.12);
        color: #14562d;
        border: 1px solid rgba(30,120,60,0.2);
    }
    .dashboard-panel {
        background: linear-gradient(180deg, #f8fff9 0%, #f7fbff 100%);
        border: 1px solid rgba(27, 94, 32, 0.12);
        border-radius: 18px;
        padding: 1rem 1rem 0.4rem 1rem;
        box-shadow: 0 10px 22px rgba(40,80,55,0.08);
        margin-bottom: 1rem;
    }
    .stButton > button {
        background: linear-gradient(180deg, #2e7d32 0%, #1b5e20 100%);
        color: white;
        font-size: 1rem;
        font-weight: 700;
        width: 100%;
        border-radius: 12px;
        border: 0;
        padding: 0.8rem 1rem;
        box-shadow: 0 8px 18px rgba(46,125,50,0.18);
    }
    .stButton > button:hover {
        background: linear-gradient(180deg, #388e3c 0%, #256d2a 100%);
        box-shadow: 0 10px 20px rgba(46,125,50,0.22);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f3fbf4 0%, #edf8ff 100%);
    }
    [data-testid="stMetricValue"] {
        font-size: 1.7rem;
        font-weight: 800;
        color: #114d2a;
    }
    [data-testid="stMetricLabel"] {
        color: #3d5c4d;
        font-weight: 700;
    }
    .section-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        background: rgba(40, 120, 60, 0.10);
        border: 1px solid rgba(40, 120, 60, 0.18);
        color: #175c2a;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    .report-box {
        background: linear-gradient(135deg, #f7fff8 0%, #f2f9ff 100%);
        border-left: 5px solid #2e7d32;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        color: #173d28;
        margin-top: 0.5rem;
    }
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_trained_model():
    """Load the checkpoint only after a real training run has created it."""
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        class_names = checkpoint["class_names"]
        model = get_crop_model(checkpoint.get("num_classes", len(class_names)))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, class_names, None
    except (KeyError, RuntimeError, OSError) as error:
        return None, None, str(error)


def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_rgb = image.convert("RGB")
    return image_rgb, transform(image_rgb).unsqueeze(0)


def calculate_weather_risk(humidity: float, temperature: float):
    if humidity > 80 and 20 <= temperature <= 30:
        return "high"
    if humidity > 65:
        return "medium"
    return "low"


def summarize_field_status(predicted_class: str, confidence: float, humidity: float, temperature: float):
    """Turn prediction data into a practical field assessment for farmers."""
    weather_risk = calculate_weather_risk(humidity, temperature)
    is_healthy = "Healthy" in predicted_class

    if is_healthy:
        health_score = int(confidence * 100)
        health_label = "Healthy crop"
        urgency = "Low"
        recommended_action = "Continue regular monitoring and maintain standard crop care."
    else:
        health_score = max(0, int((1.0 - confidence) * 100))
        health_label = "Needs attention"
        urgency = "High" if (confidence > 0.8 or weather_risk == "high") else "Medium"
        recommended_action = (
            "Inspect the affected leaf area immediately, remove infected plant parts, "
            "and consult a local agricultural expert for treatment advice."
        )

    return {
        "weather_risk": weather_risk,
        "health_score": health_score,
        "health_label": health_label,
        "urgency": urgency,
        "recommended_action": recommended_action,
        "summary": (
            f"{predicted_class} is the model output with {confidence * 100:.1f}% confidence. "
            f"Current weather conditions sit in the {weather_risk} risk band."
        ),
    }


def get_crop_context(predicted_class: str):
    """Return farmer-friendly crop, season, and planning data from the prediction."""
    raw_crop_name = predicted_class.split("___")[0]
    crop_name = raw_crop_name.replace("_", " ").title()
    month = datetime.now().month
    if month in (6, 7, 8, 9, 10):
        season = "Kharif"
    elif month in (11, 12, 1, 2, 3):
        season = "Rabi"
    else:
        season = "Zaid"

    stage_map = {
        "Apple": "Leaf and fruit development stage",
        "Tomato": "Vegetative to fruit-setting stage",
        "Corn": "Vegetative to tasseling stage",
    }
    crop_stage = stage_map.get(crop_name, "Field monitoring stage")

    planning_note = {
        "Apple": "Keep canopy airy and inspect fruit surfaces frequently.",
        "Tomato": "Check leaf joints and lower canopy for disease spread.",
        "Corn": "Monitor leaf health and avoid excess humidity around the canopy.",
    }.get(crop_name, "Continue regular field scouting and keep records.")

    return {
        "crop_name": crop_name,
        "season": season,
        "stage": crop_stage,
        "planning_note": planning_note,
    }


def build_diagnosis_report(predicted_class: str, confidence: float, humidity: float, temperature: float, language: str):
    """Create a readable summary report for farmers to share or save."""
    status = summarize_field_status(predicted_class, confidence, humidity, temperature)
    crop_context = get_crop_context(predicted_class)
    tasks = []
    for key in [
        "checklist_inspect",
        "checklist_remove",
        "checklist_irrigation",
        "checklist_expert",
    ]:
        if st.session_state.get(key):
            tasks.append(get_translation(language, key))

    lines = [
        "KrishiKavach Diagnosis Report",
        "===============================",
        f"Crop: {crop_context['crop_name']}",
        f"Prediction: {predicted_class}",
        f"Confidence: {confidence * 100:.1f}%",
        f"Season: {crop_context['season']}",
        f"Crop stage: {crop_context['stage']}",
        f"Weather risk: {status['weather_risk'].upper()}",
        f"Temperature: {temperature:.1f}°C",
        f"Humidity: {humidity:.0f}%",
        f"Field health score: {status['health_score']}/100",
        f"Urgency: {status['urgency']}",
        f"Recommended action: {status['recommended_action']}",
        f"Planning note: {crop_context['planning_note']}",
        "",
        "Checklist completed:",
    ]
    if tasks:
        for task in tasks:
            lines.append(f"- {task}")
    else:
        lines.append("- No checklist actions were marked complete.")

    return "\n".join(lines)


def render_dashboard_panel(predicted_class: str, confidence: float, humidity: float, temperature: float):
    """Render a concise diagnostics dashboard for farmers."""
    field_status = summarize_field_status(predicted_class, confidence, humidity, temperature)
    risk_score = {"low": 35, "medium": 65, "high": 90}[field_status["weather_risk"]]
    chart_data = pd.DataFrame(
        {
            "Metrics": ["Model confidence", "Field health", "Weather risk"],
            "Score": [min(100.0, confidence * 100.0), float(field_status["health_score"]), float(risk_score)],
        }
    ).set_index("Metrics")

    st.markdown('<div class="dashboard-panel">', unsafe_allow_html=True)
    st.markdown(f"<div class='status-pill'>{field_status['urgency']} priority</div>", unsafe_allow_html=True)
    st.write(f"Prediction: {predicted_class}")
    st.progress(min_value=0, max_value=100, value=confidence * 100)
    st.bar_chart(chart_data, height=220, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def record_analysis_history(predicted_class: str, confidence: float, weather_risk: str):
    """Keep a short history of recent scans in session memory."""
    history = st.session_state.setdefault("analysis_history", [])
    history.insert(
        0,
        {
            "class": predicted_class,
            "confidence": round(confidence * 100, 1),
            "risk": weather_risk,
            "timestamp": datetime.now().strftime("%H:%M"),
        },
    )
    st.session_state["analysis_history"] = history[:5]


def render_action_checklist(language: str):
    """Display a short actionable checklist for the farmer after diagnosis."""
    st.subheader(get_translation(language, "action_checklist"))
    tasks = [
        "checklist_inspect",
        "checklist_remove",
        "checklist_irrigation",
        "checklist_expert",
    ]
    for task_key in tasks:
        st.checkbox(get_translation(language, task_key), key=f"{task_key}")


def fetch_current_weather(latitude: float, longitude: float) -> dict:
    """Fetch current local weather without an API key from Open-Meteo."""
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m",
            "timezone": "auto",
        },
        timeout=10,
    )
    response.raise_for_status()
    current = response.json()["current"]
    return {
        "temperature": float(current["temperature_2m"]),
        "humidity": float(current["relative_humidity_2m"]),
    }


def request_location_weather(language: str):
    """Ask the browser for permission, then copy weather values into form state."""
    location = get_geolocation(component_key="krishidrishti_location")
    if not location:
        st.info(get_translation(language, "location_waiting"))
        return
    if "error" in location:
        st.warning(f"{get_translation(language, 'location_error')}: {location['error']['message']}")
        st.session_state.pop("location_request_active", None)
        return

    try:
        coords = location["coords"]
        weather = fetch_current_weather(coords["latitude"], coords["longitude"])
        st.session_state["temperature"] = weather["temperature"]
        st.session_state["humidity"] = weather["humidity"]
        st.session_state["location_weather_message"] = (
            f"{get_translation(language, 'location_success')}: "
            f"{weather['temperature']:.1f}°C, {weather['humidity']:.0f}%"
        )
        st.session_state.pop("location_request_active", None)
        st.rerun()
    except (KeyError, TypeError, requests.RequestException) as error:
        st.warning(f"{get_translation(language, 'weather_fetch_error')}: {error}")
        st.session_state.pop("location_request_active", None)


def show_advisory(predicted_class: str, language: str):
    advisory = CROP_KNOWLEDGE_BASE.get(predicted_class)
    if advisory is None:
        st.info("No curated advisory is available for this class yet.")
        return

    st.subheader(get_translation(language, "advisory_header"))
    st.info(get_translation(language, "advisory_subnote"))
    st.markdown(f"**{get_translation(language, 'status_label')}:** {advisory['status']}")
    st.markdown(f"**{get_translation(language, 'symptoms_label')}:** {advisory['symptoms']}")
    st.markdown(f"**{get_translation(language, 'prevention_label')}:**")
    for action in advisory["preventive_actions"]:
        st.markdown(f"- {action}")
    st.markdown(f"**{get_translation(language, 'treatment_label')}:** {advisory['treatment_guidance']}")
    st.warning(f"⚠️ **{get_translation(language, 'safety_note_label')}:** {advisory['safety_note']}")


def main():
    st.markdown("<h1 class='main-header'>🌾 KrishiKavach</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-panel">
            <div class="section-badge">Smart crop care</div>
            <div style="font-size:1.1rem; color:#134a2d; font-weight:600;">
                Explainable crop diagnosis, weather-aware guidance, and field action planning for farmers.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    language = st.selectbox(
        "🌐 Language / भाषा Selection", list(LANGUAGES), format_func=LANGUAGES.get
    )
    st.markdown(
        f"<h4 class='sub-header'>{get_translation(language, 'app_subtitle')}</h4>",
        unsafe_allow_html=True,
    )

    model, class_names, model_error = load_trained_model()
    if model_error:
        st.error(f"Saved model could not be loaded: {model_error}")
        st.stop()
    if model is None:
        st.warning(get_translation(language, "no_model_warning"))
        st.markdown("#### Complete these steps to enable prediction")
        st.code(
            "dataset/\n"
            "  Apple___Healthy/\n"
            "  Apple___Black_rot/\n"
            "  Tomato___Late_blight/\n"
            "  Corn___Common_rust/\n\n"
            "py train.py\n"
            "py -m streamlit run app.py --server.port 8502",
            language="powershell",
        )
        st.info(
            "Add labelled images to each class folder first. Training creates "
            "crop_disease_model.pth; then restart Streamlit."
        )
        st.caption("The app intentionally does not make a random or untrained prediction.")
        st.stop()

    st.subheader(get_translation(language, "weather_section"))
    st.caption(get_translation(language, "location_privacy"))
    if st.button(get_translation(language, "use_location_btn")):
        st.session_state["location_request_active"] = True
    if st.session_state.get("location_request_active"):
        request_location_weather(language)
    if st.session_state.get("location_weather_message"):
        st.success(st.session_state["location_weather_message"])

    weather_summary = st.columns(3)
    with weather_summary[0]:
        st.markdown('<div class="feature-card"><b>🌦️ Weather</b><br>Live risk check for crop infection conditions.</div>', unsafe_allow_html=True)
    with weather_summary[1]:
        st.markdown('<div class="feature-card"><b>🧠 AI Insight</b><br>Explainable prediction with Grad-CAM heatmap.</div>', unsafe_allow_html=True)
    with weather_summary[2]:
        st.markdown('<div class="feature-card"><b>📋 Action Plan</b><br>Recommended steps and monitoring checklist.</div>', unsafe_allow_html=True)

    weather_left, weather_right = st.columns(2)
    temperature = weather_left.number_input(
        get_translation(language, "temp_label"), 0.0, 50.0, 25.0, 0.5, key="temperature"
    )
    humidity = weather_right.number_input(
        get_translation(language, "humidity_label"), 0.0, 100.0, 85.0, 1.0, key="humidity"
    )

    st.subheader(get_translation(language, "upload_section"))
    uploaded_file = st.file_uploader(
        get_translation(language, "upload_help"), type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is None:
        st.divider()
        st.markdown(f"### 🛡️ {get_translation(language, 'disclaimer_header')}")
        st.caption(get_translation(language, "disclaimer_body"))
        return

    try:
        uploaded_image = Image.open(uploaded_file)
    except (OSError, ValueError):
        st.error("The uploaded file is not a readable image.")
        return

    st.image(uploaded_image, caption="Uploaded leaf image", use_container_width=True)
    st.markdown('<div class="report-box">Upload a healthy crop photo or a leaf showing symptoms to get AI-based diagnosis and guidance.</div>', unsafe_allow_html=True)
    if not st.button(get_translation(language, "analyze_btn")):
        return

    with st.spinner("Analyzing image and generating visual evidence..."):
        image_rgb, input_tensor = preprocess_image(uploaded_image)
        with torch.no_grad():
            scores = model(input_tensor)
            probabilities = torch.softmax(scores, dim=1)
        confidence, prediction_index = torch.max(probabilities, 1)
        predicted_class = class_names[prediction_index.item()]

        grad_cam = GradCAM(model, model.target_layer)
        heatmap, _ = grad_cam.generate_heatmap(input_tensor, prediction_index.item())
        grad_cam.close()
        overlay = apply_heatmap_overlay(np.array(image_rgb.resize((224, 224))), heatmap)

    st.divider()
    st.markdown('<div class="section-badge">Diagnosis summary</div>', unsafe_allow_html=True)
    st.subheader(get_translation(language, "pred_results_header"))
    result_left, result_right = st.columns(2)
    result_left.success(f"**{get_translation(language, 'pred_disease')}:** {predicted_class}")
    result_right.info(f"**{get_translation(language, 'confidence')}:** {confidence.item() * 100:.2f}%")

    field_status = summarize_field_status(predicted_class, confidence.item(), humidity, temperature)
    risk = field_status["weather_risk"]
    crop_context = get_crop_context(predicted_class)

    render_dashboard_panel(predicted_class, confidence.item(), humidity, temperature)

    st.subheader(get_translation(language, "field_summary"))
    field_health, field_weather, field_urgency = st.columns(3)
    with field_health:
        st.metric(get_translation(language, "field_health_score"), f"{field_status['health_score']}/100", field_status["health_label"])
    with field_weather:
        st.metric(get_translation(language, "weather_pressure"), risk.upper(), f"{temperature}°C / {humidity}%")
    with field_urgency:
        st.metric(get_translation(language, "urgency"), field_status["urgency"], "")
    st.info(f"**{get_translation(language, 'recommended_action')}:** {field_status['recommended_action']}")
    st.caption(field_status["summary"])
    record_analysis_history(predicted_class, confidence.item(), risk)

    st.subheader(get_translation(language, "season_planner"))
    season_col1, season_col2, season_col3 = st.columns(3)
    with season_col1:
        st.metric(get_translation(language, "current_season"), crop_context["season"])
    with season_col2:
        st.metric(get_translation(language, "crop_stage"), crop_context["stage"])
    with season_col3:
        st.metric(get_translation(language, "best_window"), get_translation(language, "monitoring_window"))
    st.caption(crop_context["planning_note"])

    st.subheader(get_translation(language, "gradcam_header"))
    st.image(overlay, caption="Grad-CAM model evidence heatmap", width=360)
    st.caption(f"💡 {get_translation(language, 'gradcam_note')}")

    st.divider()
    show_advisory(predicted_class, language)
    render_action_checklist(language)

    report_text = build_diagnosis_report(predicted_class, confidence.item(), humidity, temperature, language)
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.download_button(
        get_translation(language, "download_report"),
        data=report_text,
        file_name=f"{predicted_class.replace('___', '_')}_report.txt",
        mime="text/plain",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader(get_translation(language, "weather_alert_header"))
    if risk == "high":
        st.error(f"🔴 **{get_translation(language, 'risk_high')}**\n\n{get_translation(language, 'risk_high_msg')}")
    elif risk == "medium":
        st.warning(f"🟠 **{get_translation(language, 'risk_med')}**\n\n{get_translation(language, 'risk_med_msg')}")
    else:
        st.success(f"🟢 **{get_translation(language, 'risk_low')}**\n\n{get_translation(language, 'risk_low_msg')}")

    st.divider()
    st.markdown(f"### 🛡️ {get_translation(language, 'disclaimer_header')}")
    st.caption(get_translation(language, "disclaimer_body"))

    with st.sidebar:
        st.subheader(get_translation(language, "recent_scans"))
        history = st.session_state.get("analysis_history", [])
        if not history:
            st.info(get_translation(language, "recent_scans_empty"))
        for item in history:
            badge = "⚠️" if item["risk"] != "low" else "✅"
            st.markdown(
                f"{badge} **{item['class']}**\n"
                f"{item['confidence']}% · {item['risk']} risk · {item['timestamp']}"
            )
        if history:
            st.markdown("<hr style='margin-top:1rem; margin-bottom:0.7rem;' />", unsafe_allow_html=True)
            st.caption("KrishiKavach • field intelligence")


if __name__ == "__main__":
    main()
