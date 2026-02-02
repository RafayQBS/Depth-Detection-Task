import cv2
import torch
import numpy as np
import threading
from queue import Queue
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2

# ===============================
# CONFIG & HYPERPARAMETERS
# ===============================
YOLO_MODEL_URL = "https://ai-public-videos.s3.us-east-2.amazonaws.com/weights/obb.pt"
VIDEO_URL = "https://ai-public-videos.s3.us-east-2.amazonaws.com/Raw+Videos/Navirox/sorted/accident_left_2.mp4"

# Performance & Display
SCALE_FACTOR = 0.7
INFERENCE_RES = (int(400 * SCALE_FACTOR), int(700 * SCALE_FACTOR))  
DISPLAY_WIDTH, DISPLAY_HEIGHT = 400, 700
FPS = 30  # Targeted output FPS

# Depth Calibration
ALPHA = 0.15         # Temporal smoothing (lower = smoother)
METRIC_FACTOR = 200.0 # Adjust this to calibrate real-world meters
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Async Queues
input_queue = Queue(maxsize=5)
stop_event = threading.Event()

# State for temporal smoothing
depth_history = {}

# ===============================
# THREAD: ASYNC VIDEO READER
# ===============================
def frame_reader(url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("[ERROR] Could not open video stream.")
        stop_event.set()
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            stop_event.set()
            break
        
        # Pre-resize to model native resolution to save VRAM and transfer time
        resized = cv2.resize(frame, INFERENCE_RES)
        if not input_queue.full():
            input_queue.put(resized)
    cap.release()

# ===============================
# MODEL INITIALIZATION
# ===============================
print(f"[INFO] Initializing Models on {DEVICE}...")

# Load YOLO
yolo_model = YOLO(YOLO_MODEL_URL)

# Load DepthAnythingV2 (ViT-B)
model_config = {
    "encoder": "vitb",
    "features": 128,
    "out_channels": [96, 192, 384, 768]
}
depth_model = DepthAnythingV2(**model_config)
depth_model.load_state_dict(torch.load("depth_anything_v2_vitb.pth", map_location="cpu"))
depth_model = depth_model.to(DEVICE).eval()

# ===============================
# MAIN INFERENCE & RENDERING
# ===============================
def run_pipeline():
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter("optimized_maritime_depth.mp4", fourcc, FPS, (DISPLAY_WIDTH * 2, DISPLAY_HEIGHT))

    print("[INFO] Inference Running. Press 'q' to quit.")

    while not stop_event.is_set():
        if input_queue.empty():
            continue
            
        frame = input_queue.get()
        
        # 1. AI Inference
        with torch.no_grad():
            # YOLO OBB Detection
            results = yolo_model(frame, conf=0.3, iou=0.5, device=0, verbose=False)
            # Depth Estimation
            depth_map = depth_model.infer_image(frame)

        # 2. Distance Normalization
        # We normalize to 0-1 range locally for visualization
        depth_min, depth_max = depth_map.min(), depth_map.max()
        depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-6)
        
        # Create Colormap
        depth_vis = (depth_norm * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_PLASMA)

        annotated_frame = frame.copy()

        # 3. Object Processing with Temporal Smoothing & Metric Correction
        for i, (box, cls) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.cls)):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)
            class_name = yolo_model.names[class_id]
            
            if class_name == 'person': continue # Focusing on boats

            # ROI Median Sampling (Robust against waves/spray)
            h, w = y2 - y1, x2 - x1
            roi = depth_map[y1 + h//4:y2 - h//4, x1 + w//4:x2 - w//4]
            
            if roi.size > 0:
                # INVERSE DEPTH LOGIC: 
                # Model outputs high values for close things. 
                # We invert it so small disparities = large distances.
                raw_val = np.median(roi)
                
                # Metric calculation: Distance is inversely proportional to disparity
                # We use (1.0 - normalized_value) to ensure horizon (0) = far (1)
                norm_val = (raw_val - depth_min) / (depth_max - depth_min + 1e-6)
                dist_metric = METRIC_FACTOR / (norm_val + 0.01) 

                # Temporal Smoothing (EMA)
                # Note: Using box index 'i' as temporary ID. 
                if i not in depth_history:
                    depth_history[i] = dist_metric
                else:
                    depth_history[i] = (ALPHA * dist_metric) + ((1 - ALPHA) * depth_history[i])

                smooth_dist = depth_history[i]

                # 4. Rendering
                color = (0, 255, 0)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Label with distance
                label = f"{class_name}: {smooth_dist:.1f}m"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 5. Side-by-Side Assembly
        res_frame = cv2.resize(annotated_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        dep_frame = cv2.resize(depth_colormap, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        combined = np.hstack((res_frame, dep_frame))

        cv2.imshow("Maritime Depth Pipeline", combined)
        out_video.write(combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()

    out_video.release()
    cv2.destroyAllWindows()

# ===============================
# EXECUTION ENTRY
# ===============================
if __name__ == "__main__":
    # Start the async reader
    reader_thread = threading.Thread(target=frame_reader, args=(VIDEO_URL,), daemon=True)
    reader_thread.start()
    
    try:
        run_pipeline()
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        print("[INFO] Pipeline shut down successfully.")