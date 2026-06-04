import streamlit as st
import cv2
from ultralytics import YOLO
import time
from datetime import datetime
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="PPE Safety Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- 3. SESSION STATE ---
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

# --- 4. SIDEBAR CONTROLS ---
st.sidebar.title("NAVIGATOR")
app_mode = st.sidebar.radio("Select View:", ["Dashboard (Admin)", "Full Screen (Site View)"])
st.sidebar.markdown("---")
st.sidebar.header("Global Controls")
run_detection = st.sidebar.toggle("🔴 START SYSTEM", value=False)
enable_audio = st.sidebar.checkbox("Enable Notifications", value=True)

# --- 5. DEFINE LAYOUT (Runs EXACTLY ONCE) ---
video_placeholder = None
queue_placeholder = None
conf_threshold = 0.50

if app_mode == "Dashboard (Admin)":
    st.title("🚧 Construction Site Safety Monitor")
    col1, col2 = st.columns([0.65, 0.35])

    with col1:
        st.subheader("Live Feed")
        video_placeholder = st.empty() 
        
        st.markdown("---")
        st.markdown("### 🎚️ Sensitivity Settings")
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.45, 0.05, key="main_conf_slider")

    with col2:
        st.subheader("📋 Verification Queue")
        queue_placeholder = st.empty()

elif app_mode == "Full Screen (Site View)":
    st.markdown("## 🔴 LIVE SITE MONITORING - DO NOT ENTER WITHOUT GEAR")
    video_placeholder = st.empty()
    conf_threshold = 0.50

# --- 6. UI UPDATER FUNCTIONS ---
def update_sidebar_passive():
    """Updates sidebar without buttons (Safe for Loop)"""
    if app_mode == "Dashboard (Admin)" and queue_placeholder is not None:
        with queue_placeholder.container():
            if len(st.session_state.alerts) == 0:
                st.info("No active violations.")
            else:
                st.write(f"**Pending Reviews: {len(st.session_state.alerts)}**")
                for alert in reversed(st.session_state.alerts):
                    with st.expander(f"🚨 {alert['type']} ({alert['time']})", expanded=False):
                        st.image(alert['image'], channels="BGR", use_container_width=True)

def update_sidebar_active():
    """Updates sidebar WITH buttons (Only when Stopped)"""
    if app_mode == "Dashboard (Admin)" and queue_placeholder is not None:
        with queue_placeholder.container():
            if len(st.session_state.alerts) == 0:
                st.success("All clear! No pending violations.")
            else:
                st.warning(f"**Action Required: {len(st.session_state.alerts)} Alerts**")
                for alert in reversed(st.session_state.alerts):
                    with st.expander(f"🚨 {alert['type']} ({alert['time']})", expanded=True):
                        st.image(alert['image'], channels="BGR", use_container_width=True)
                        c1, c2 = st.columns(2)
                        
                        if c1.button("✅ Log", key=f"log_{alert['id']}"):
                            st.toast(f"Logged: {alert['type']}")
                            st.session_state.alerts = [a for a in st.session_state.alerts if a['id'] != alert['id']]
                            st.rerun()
                            
                        if c2.button("❌ Ignore", key=f"ign_{alert['id']}"):
                            st.toast("Dismissed")
                            st.session_state.alerts = [a for a in st.session_state.alerts if a['id'] != alert['id']]
                            st.rerun()

# --- 7. MAIN LOGIC ---

if run_detection:
    cap = cv2.VideoCapture(0) # Use 0 for default camera; replace with video file path if needed. 
    last_snapshot_time = 0 
    snapshot_cooldown = 2.5 
    
    # Render Initial UI
    update_sidebar_passive()

    try:
        while run_detection:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera Error: Check connection.")
                break

            # A. Inference
            results = model(frame, conf=0.35, verbose=False) # Run low sensitivity to catch everything first
            annotated_frame = frame.copy() # We draw manually to avoid clutter

            # B. SMART FILTERING LOGIC
            violation_type = None
            current_violation_box = None
            
            # Draw valid boxes manually
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                
                # RULE: Strict Thresholds for Violations
                is_violation = label in ['NO-Hardhat', 'NO-Safety Vest']
                
                # If it's a "NO-" class, require HIGHER confidence (e.g. 0.75) to avoid false alarms
                if is_violation and conf < 0.75:
                    continue # Skip this weak detection (fixes your 0.71 error)
                
                # If it's a normal class, use the slider threshold
                if not is_violation and conf < conf_threshold:
                    continue

                # Draw Box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                if is_violation:
                    color = (0, 0, 255) # Red for danger
                    violation_type = label
                    current_violation_box = box
                else:
                    color = (0, 255, 0) # Green for safe
                
            

            # C. Handle Violation
            if violation_type:
                h, w, _ = annotated_frame.shape
                cv2.rectangle(annotated_frame, (0, 0), (w, h), (0, 0, 255), 30)
                cv2.rectangle(annotated_frame, (0, 0), (w, 80), (0, 0, 255), -1)
                cv2.putText(annotated_frame, f"WARNING: {violation_type}!", (20, 55), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

                if app_mode == "Dashboard (Admin)" and enable_audio:
                    current_time = time.time()
                    if current_time - last_snapshot_time > snapshot_cooldown:
                        new_alert = {
                            "id": str(time.time()), 
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "type": violation_type,
                            "image": frame.copy(), 
                            "conf": float(current_violation_box.conf[0])
                        }
                        st.session_state.alerts.append(new_alert)
                        st.toast(f"⚠️ {violation_type} Detected")
                        update_sidebar_passive()
                        last_snapshot_time = current_time

            # D. Update Video
            if video_placeholder is not None:
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    except Exception as e:
        st.error(f"System Error: {e}")
    finally:
        # Ensures camera is released even if you stop the script
        cap.release()

else:
    if app_mode == "Dashboard (Admin)":
        
        update_sidebar_active()
    else:
        
        st.info("System Paused.")


    # Check For UI fix also
    #Have to work on multi-camera setup as well.
