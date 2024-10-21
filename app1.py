import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from optimized import OptimizedOpticalFlowTracker
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Styling and Page Config
st.set_page_config(page_title="Optical Flow Tracker Pro", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for dark theme and premium look
st.markdown("""
<style>
    body {
        color: #E0E0E0;
        background-color: #1E1E1E;
    }
    .stApp {
        background-color: #1E1E1E;
    }
    .main {
        background-color: #2D2D2D;
        padding: 2rem;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stSelectbox, .stNumberInput {
        margin-bottom: 1rem;
        background-color: #3D3D3D;
        color: #E0E0E0;
        border-radius: 5px;
    }
    .css-145kmo2 {
        color: #E0E0E0;
    }
    .css-1d391kg {
        background-color: #3D3D3D;
    }
    h1, h2, h3 {
        color: #4CAF50;
    }
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    .streamlit-expanderHeader {
        background-color: #3D3D3D;
        color: #E0E0E0;
    }
    .css-1adrfps {
        background-color: #2D2D2D;
    }
</style>
""", unsafe_allow_html=True)

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.tracker = OptimizedOpticalFlowTracker()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed_frame = self.tracker.process_frame(img, 30)  # Assume 30 FPS for live video
        return processed_frame

def main():
    st.title("🎥 Optical Flow Tracker Pro")

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        input_option = st.radio("Select Input Source", ["Upload Video", "Record from Camera"])

        if input_option == "Upload Video":
            uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
        else:
            st.info("Click 'Start' below to begin camera recording")

        # Video processing options
        max_frames = st.number_input("Max Frames to Process", value=100, min_value=1)
        detection_interval = st.number_input("Detection Interval", value=5, min_value=1)

        st.markdown("---")
        st.subheader("About")
        st.info("Optical Flow Tracker Pro uses advanced algorithms to process videos. "
                "Upload a video or record live from your camera, and adjust settings to customize processing.")

    # Main content area
    col1, col2 = st.columns(2)

    with col1:
        st.header("Original Video")
        original_video_container = st.empty()
        if input_option == "Upload Video" and uploaded_file is not None:
            # Display uploaded video
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            original_video_container.video(tfile.name)
        elif input_option == "Record from Camera":
            # Live video recording
            ctx = webrtc_streamer(
                key="camera",
                video_processor_factory=VideoProcessor,
                rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            )

    with col2:
        st.header("Processed Video")
        processed_video_container = st.empty()

    # Process Video Button
    if st.button("Process Video", key="process_button"):
        if input_option == "Upload Video" and uploaded_file is not None:
            # Process uploaded video
            tracker = OptimizedOpticalFlowTracker()
            tracker.detection_interval = detection_interval

            cap = cv2.VideoCapture(tfile.name)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            frame_count = 0
            processed_frames = []

            progress_bar = st.progress(0)
            status_text = st.empty()

            while True:
                ret, frame = cap.read()
                if not ret or frame_count >= max_frames:
                    break

                processed_frame = tracker.process_frame(frame, fps)
                processed_frames.append(processed_frame)

                # Update progress
                progress = int((frame_count / max_frames) * 100)
                progress_bar.progress(progress)
                status_text.text(f"Processing: {progress}% complete")

                frame_count += 1

            cap.release()

            # Save processed video
            output_file = "processed_video.mp4"
            out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (640, 640))
            for frame in processed_frames:
                out.write(frame)
            out.release()

            # Display processed video
            processed_video_container.video(output_file)

            # Provide download link
            st.success("Video processing complete!")
            with open(output_file, "rb") as file:
                btn = st.download_button(
                    label="Download processed video",
                    data=file,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )

            # Clean up the temporary file
            os.unlink(tfile.name)
        elif input_option == "Record from Camera":
            st.info("Processing live video. The processed frames are displayed in real-time above.")

if __name__ == "__main__":
    main()