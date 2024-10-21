import streamlit as st
import cv2
import tempfile
from run import OptimizedOpticalFlowTracker
import os
import time

# Initialize tracker
tracker = OptimizedOpticalFlowTracker()

# Streamlit title
st.title("Optimized Optical Flow Tracker")

# Sidebar options
st.sidebar.title("Options")

# Option to upload a video or use webcam
input_source = st.sidebar.radio("Choose input source:", ('Upload Video', 'Use Webcam'))

# Initialize variables
input_file = None
video_file_path = None
output_file_path = None

def process_video(input_path, output_path):
    # Open the input video file
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        st.error(f"Error: Could not open input source: {input_path}")
        return None

    # Get input video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Set FourCC codec to ensure mp4 format with H264 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create a VideoWriter object using mp4v codec and save as .mp4
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_output.name

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate FPS
        elapsed_time = time.time() - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # Process the frame with the tracker
        processed_frame = tracker.process_frame(frame, current_fps)
        out.write(processed_frame)

        frame_count += 1

    cap.release()
    out.release()
    st.success(f"Video processing complete. Output saved as '{output_path}'")
    return output_path

# Input handling based on user's choice
if input_source == 'Upload Video':
    # Video upload option
    input_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if input_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(input_file.read())
        video_file_path = tfile.name

elif input_source == 'Use Webcam':
    st.sidebar.write("Record video using webcam")
    video_file_path = "webcam_video.mp4"  # Placeholder for webcam recording

# Display input video
if video_file_path:
    st.write("### Input Video")
    st.video(video_file_path)

    # Process video on button click
    if st.button("Process Video"):
        # Set the output path for the processed video
        output_file_path = process_video(video_file_path, output_file_path)

# Display output video if available
if output_file_path and os.path.exists(output_file_path):
    st.write("### Output Video")
    st.video(output_file_path)  # Streamlit video component expects a file path
