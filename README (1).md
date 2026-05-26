#  Football Vision AI

AI-powered football analytics system using YOLOv8 + ByteTrack to detect and track players and ball in real-time, generating broadcast-style movement heatmaps.

## Demo
Upload any football video → get player tracking + heatmap instantly.

##  Features
- Real-time player + ball detection using YOLOv8
- Multi-object tracking with ByteTrack (unique ID per player)
- Movement heatmap showing field coverage patterns
- Works with any video resolution

## Tech Stack
- YOLOv8 (Ultralytics)
- ByteTrack (Supervision)
- OpenCV
- Streamlit

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

##  How It Works
1. YOLOv8 detects players and ball every frame
2. ByteTrack assigns consistent IDs across frames
3. Player positions accumulated into a heatmap
4. Heatmap overlaid on video frame using JET colormap
