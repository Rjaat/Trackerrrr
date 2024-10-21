import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time
import argparse

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
        self.class_names = 'object' #self.yolo.names  # Load class names from YOLO

        #print("Using CPU-based OpenCV for optical flow")

    def preprocess_frame(self, frame, target_size=(640, 640)):
        return cv2.resize(frame, target_size)

    def detect_objects(self, frame):
        results = self.yolo(frame, conf=0.3, iou=0.5)
        return results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else []

    def calculate_optical_flow(self, frame_gray):
        if self.prev_gray is None:
            self.prev_gray = frame_gray
            return None

        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.1, 0)
        self.prev_gray = frame_gray
        return flow

    def update_tracks(self, detections, flow):
        if len(detections) == 0:
            #print("No detections found, skipping frame.")
            return self.tracks  # Skip frame if no detections

        if self.prev_points is None or len(self.prev_points) == 0:
            # Initialize tracking points based on detections
            self.prev_points = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in detections])
            self.tracks = {tuple(point): self.track_id + i for i, point in enumerate(self.prev_points)}
            self.track_id += len(self.prev_points)
            return self.tracks

        if flow is not None:
            if flow.ndim == 3 and self.prev_points.ndim == 2 and len(self.prev_points) > 0:
                h, w, _ = flow.shape
                valid_points = (self.prev_points[:, 0] < w) & (self.prev_points[:, 1] < h)
                valid_prev_points = self.prev_points[valid_points]
                
                if len(valid_prev_points) > 0:  # Ensure there are valid points to update
                    new_points = valid_prev_points + flow[valid_prev_points[:, 1].astype(int), valid_prev_points[:, 0].astype(int), :]
                    new_tracks = {}
                    for old_point, new_point in zip(valid_prev_points, new_points):
                        old_track_id = self.tracks.get(tuple(old_point))
                        if old_track_id is not None:
                            new_tracks[tuple(new_point)] = old_track_id
                    self.tracks = new_tracks
                    self.prev_points = valid_prev_points
            else:
                print("Flow or points dimensions are not as expected or no valid points.")
                return self.tracks

        return self.tracks

    def process_frame(self, frame, fps):
        frame = self.preprocess_frame(frame)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.frame_count % self.detection_interval == 0:
            self.last_detections = self.detect_objects(frame)
        
        flow = self.calculate_optical_flow(frame_gray)
        self.tracks = self.update_tracks(self.last_detections, flow)
        
        frame = self.visualize(frame, self.tracks, flow, fps)
        
        self.frame_count += 1
        return frame

    def visualize(self, frame, tracks, flow, fps):
        # Display Track IDs and Class Names on Detected Objects
        for detection in self.last_detections:
            if len(detection) >= 4:  # Ensure there are enough elements
                x1, y1, x2, y2 = map(int, detection[:4])  # Get bounding box coordinates
                cls = int(detection[5]) if len(detection) >= 6 else 0  # Get class index
                conf = detection[4] if len(detection) >= 5 else 1.0  # Get confidence

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Display class name and ID if confidence is high enough
                if conf > 0.3:  # Display only if confidence is above threshold
                    class_name = 'object'
                    cv2.putText(frame, f"{class_name} ID: {tracks.get((x1 + x2) // 2, 'N/A')}", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw Flow Lines (optional)
        if flow is not None:
            step = 16
            h, w = flow.shape[:2]
            y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
            fx, fy = flow[y, x].T
            lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
            lines = np.int32(lines + 0.5)
            cv2.polylines(frame, lines, 0, (0, 255, 255))

        # Display FPS count
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
    out = cv2.VideoWriter(output_file, fourcc, fps, (640, 640))  # Note the output size

    frame_count = 0
    start_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate FPS
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

            processed_frame = tracker.process_frame(frame, current_fps)
            out.write(processed_frame)

            frame_count += 1
            #if frame_count % 30 == 0:

                #print(f"Processed {frame_count} frames. FPS: {current_fps:.2f}")

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
