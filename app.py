# app.py
import streamlit as st
import cv2
import pandas as pd
import time
from src import detect_emotion
get_emotion = detect_emotion.get_emotion

st.set_page_config(page_title="AI Mood Mirror", layout="wide")
st.title("😊 AI Mood Mirror — Real-Time Emotion Detector")
st.write("A lightweight, robust demo. Works with DeepFace / fer if installed, otherwise uses a fallback.")

# left: camera, right: controls & logs
col1, col2 = st.columns([2, 1])

with col2:
    st.header("Controls")
    start = st.button("Start Camera")
    stop = st.button("Stop Camera")
    save_log = st.button("Save Mood Log (CSV)")

    # Backend selection: 'auto' uses the module-detected backend, otherwise override
    backend_choice = st.selectbox(
        "Backend (choose 'auto' to use available backends)",
        ['auto', 'deepface', 'fer', 'mediapipe', 'none'],
        index=0,
    )
    if backend_choice != 'auto':
        detect_emotion.USE_BACKEND = backend_choice

    st.markdown(f"**Active backend:** `{detect_emotion.USE_BACKEND}`")

    st.markdown("**Detected Emotion History**")
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    st.dataframe(pd.DataFrame(st.session_state['history'], columns=["time", "emotion", "confidence"]).tail(10))

with col1:
    frame_window = st.image([])

# camera init
if 'cam' not in st.session_state:
    st.session_state['cam'] = None

if start:
    # open camera
    if st.session_state['cam'] is None:
        st.session_state['cam'] = cv2.VideoCapture(0)
        time.sleep(1.0)

if stop:
    if st.session_state['cam'] is not None:
        st.session_state['cam'].release()
        st.session_state['cam'] = None

# main loop
if st.session_state['cam'] is not None:
    cam = st.session_state['cam']
    ret, frame = cam.read()
    if not ret:
        st.warning("Can't read frame from camera. Make sure another app isn't using the camera.")
    else:
        # flip for mirror view
        frame = cv2.flip(frame, 1)
        # detect
        emotion, confidence = get_emotion(frame)
        # show frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb, channels='RGB')

        # display detected
        st.markdown(f"### Detected emotion: **{(emotion or 'unknown').upper()}**")
        st.write(f"Confidence: {confidence:.2f}")

        # add to history
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        st.session_state['history'].append([now, emotion, float(confidence)])

# Save history CSV if requested
if save_log:
    df = pd.DataFrame(st.session_state.get('history', []), columns=["time", "emotion", "confidence"])
    if not df.empty:
        path = "mood_log.csv"
        df.to_csv(path, index=False)
        st.success(f"Saved {len(df)} rows to {path}")
    else:
        st.info("No history to save yet.")

