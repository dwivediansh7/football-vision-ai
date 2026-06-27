import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import tempfile
import imageio
import imageio.v2 as iio

st.set_page_config(page_title="Football Vision AI", layout="wide")
st.title("Football Vision AI")
st.markdown("Upload a football clip to track players and generate movement heatmap.")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()
uploaded = st.file_uploader("Upload Football Video", type=["mp4", "avi", "mov"])

if uploaded:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded.read())

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    heatmap = np.zeros((h, w), dtype=np.float32)
    frame_count = 0
    MAX_FRAMES = 150
    last_frame = None
    annotated_frames = []

    progress = st.progress(0, "Processing video...")

    while cap.isOpened() and frame_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (w, h))
        last_frame = frame.copy()

        results = model(frame, classes=[0, 32], verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        for box in detections.xyxy:
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            if 0 <= cx < w and 0 <= cy < h:
                cv2.circle(heatmap, (cx, cy), 25, 1, -1)

        labels = [f"P{tid}" for tid in detections.tracker_id] if detections.tracker_id is not None else []
        annotated = box_annotator.annotate(frame.copy(), detections)
        annotated = label_annotator.annotate(annotated, detections, labels)

        annotated_frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        frame_count += 1
        progress.progress(frame_count / MAX_FRAMES)

    cap.release()

    final_path = tempfile.mktemp(suffix="_final.mp4")
    writer = imageio.get_writer(final_path, fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=1)
    for f in annotated_frames:
        writer.append_data(f)
    writer.close()

    heatmap_blur = cv2.GaussianBlur(heatmap, (51, 51), 0)
    if heatmap_blur.max() > 0:
        heatmap_norm = (heatmap_blur / heatmap_blur.max() * 255).astype(np.uint8)
    else:
        heatmap_norm = heatmap_blur.astype(np.uint8)

    heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(last_frame, 0.5, heatmap_colored, 0.5, 0)

    st.success("Processing complete!")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tracked Video")
        with open(final_path, "rb") as f:
            st.video(f.read())
    with col2:
        st.subheader("Movement Heatmap")
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    st.subheader("Stats")
    total = len(set(
        tid for tid in (detections.tracker_id if detections.tracker_id is not None else [])
    ))
    st.metric("Players Detected", total)
    st.metric("Frames Processed", frame_count)

