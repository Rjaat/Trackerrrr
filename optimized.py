import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time
import argparse
from threading import Thread

class OptimizedOpticalFlowTracker:
    def __init__(self, yolo_model='yolov8n.pt'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device for YOLO: {self.device}")
        self.yolo = YOLO(yolo_model)
        self.yolo.to(self.device)

        self.prev_gray = None
        self.prev_points = None
        self.track_id = 0
        self.tracks = {}
        self.last_detections = None
        self.frame_count = 0
        self.detection_interval = 5
        self.class_names = 'object'  # Replace with actual YOLO class names if necessary
        self.flow_thread = None
        self.flow_result = None

    def preprocess_frame(self, frame, target_size=(640, 640)):
        return cv2.resize(frame, target_size)

    def detect_objects(self, frame):
        # Use YOLO for object detection
        results = self.yolo(frame, conf=0.3, iou=0.5)
        return results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else []

    def calculate_optical_flow(self, frame_gray):
        if self.prev_gray is None:
            self.prev_gray = frame_gray
            return None
        # Efficient calculation of optical flow only around moving areas
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.1, 0)
        self.prev_gray = frame_gray
        self.flow_result = flow  # Store flow result for access in main thread
        return flow

    def start_flow_thread(self, frame_gray):
        # Run optical flow calculation in a separate thread for parallel processing
        if self.flow_thread is None or not self.flow_thread.is_alive():
            self.flow_thread = Thread(target=self.calculate_optical_flow, args=(frame_gray,))
            self.flow_thread.start()

    def update_tracks(self, detections, flow):
        if len(detections) == 0:
            return self.tracks  # Skip if no detections

        if self.prev_points is None or len(self.prev_points) == 0:
            # Initialize tracking points from detections
            self.prev_points = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in detections])
            self.tracks = {tuple(point): self.track_id + i for i, point in enumerate(self.prev_points)}
            self.track_id += len(self.prev_points)
            return self.tracks

        if flow is not None:
            h, w, _ = flow.shape
            valid_points = (self.prev_points[:, 0] < w) & (self.prev_points[:, 1] < h)
            valid_prev_points = self.prev_points[valid_points]
            if len(valid_prev_points) > 0:
                new_points = valid_prev_points + flow[valid_prev_points[:, 1].astype(int), valid_prev_points[:, 0].astype(int), :]
                new_tracks = {tuple(new_point): self.tracks.get(tuple(old_point)) for old_point, new_point in zip(valid_prev_points, new_points)}
                self.tracks = new_tracks
                self.prev_points = valid_prev_points

        return self.tracks

    def process_frame(self, frame, fps):
        frame = self.preprocess_frame(frame)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.frame_count % self.detection_interval == 0:
            self.last_detections = self.detect_objects(frame)

        if self.flow_thread and self.flow_thread.is_alive():
            self.flow_thread.join()  # Ensure optical flow is ready
        flow = self.flow_result

        self.start_flow_thread(frame_gray)  # Start new optical flow calculation in parallel

        self.tracks = self.update_tracks(self.last_detections, flow)
        self.frame_count += 1

        return self.visualize(frame, self.tracks, flow, fps)

    def visualize(self, frame, tracks, flow, fps):
        for detection in self.last_detections:
            if len(detection) >= 4:
                x1, y1, x2, y2 = map(int, detection[:4])
                cls = int(detection[5]) if len(detection) >= 6 else 0
                conf = detection[4] if len(detection) >= 5 else 1.0
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if conf > 0.3:
                    class_name = 'object'
                    track_id = tracks.get((x1 + x2) // 2, 'N/A')
                    cv2.putText(frame, f"{class_name} ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if flow is not None:
            step = 16
            h, w = flow.shape[:2]
            y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
            fx, fy = flow[y, x].T
            lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
            lines = np.int32(lines + 0.5)
            cv2.polylines(frame, lines, 0, (0, 255, 255))

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return frame

def main(input_source, output_file, max_frames=None):
    tracker = OptimizedOpticalFlowTracker()

    if input_source == '0':
        cap = cv2.VideoCapture(0)
        print("Using webcam as input source")
    else:
        cap = cv2.VideoCapture(input_source)
        print(f"Using video file: {input_source}")

    if not cap.isOpened():
        print(f"Error: Could not open input source: {input_source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (640, 640))

    frame_count = 0
    start_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

            processed_frame = tracker.process_frame(frame, current_fps)
            out.write(processed_frame)

            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        out.release()
        print(f"Video processing complete. Output saved as '{output_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized Optical Flow Tracker")
    parser.add_argument("--input", default="0", help="Input source. Use '0' for webcam or provide a path to a video file.")
    parser.add_argument("--output", default="output_video.mp4", help="Output video file name")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process")
    args = parser.parse_args()

    main(args.input, args.output, args.max_frames)
