import modal
import json
import time

# ==========================================
# 1. DEFINE CLOUD INFRASTRUCTURE
# ==========================================
import numpy as np
import cv2

def get_skin_mask(roi_bgr):
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Create a CLAHE object (clipLimit 2.0 and 8x8 grid are industry standards)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    # Merge back and convert to BGR
    limg = cv2.merge((cl, a_channel, b_channel))
    clahe_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Convert the flattened image to YCrCb
    ycrcb = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2YCrCb)
    
    # Standard human skin YCrCb strict bounds.
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    
    mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    
    # Morphological Cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return mask

class KalmanOpticalTracker:
    def __init__(self):
        # State: [x, y, dx, dy] | Measurement: [x, y]
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], 
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0], 
                                             [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        self.prev_gray = None
        self.prev_pts = None
        self.is_tracking = False
        self.last_box_size = 0

    def update_with_detector(self, center_x, center_y, box_size, frame_gray, frame_bgr, valid_x1, valid_y1, valid_x2, valid_y2):
        """YOLO sees the hand amodally. Update physics and extract skin pixels for future tracking."""
        measurement = np.array([[np.float32(center_x)], [np.float32(center_y)]])
        self.kf.correct(measurement)
        
        self.last_box_size = box_size
        self.prev_gray = frame_gray.copy()
        
        # Create a mask only inside the bounding box, and ONLY on skin pixels
        full_mask = np.zeros_like(frame_gray)
        roi_bgr = frame_bgr[valid_y1:valid_y2, valid_x1:valid_x2]
        
        # Protect against empty slices at the absolute borders of the frame
        if roi_bgr.size > 0:
            skin_mask_roi = get_skin_mask(roi_bgr)
            full_mask[valid_y1:valid_y2, valid_x1:valid_x2] = skin_mask_roi
            
            # Grab good tracking features that are strictly on the skin
            self.prev_pts = cv2.goodFeaturesToTrack(frame_gray, maxCorners=50, qualityLevel=0.1, minDistance=5, mask=full_mask)
            self.is_tracking = True
        else:
            self.is_tracking = False
            
        self.kf.predict() 

    def predict_with_flow(self, current_gray, width, height):
        """YOLO dropped a frame. Hallucinate the box using Optical Flow on previously verified skin pixels."""
        if not self.is_tracking or self.prev_pts is None:
            return None, None, None 
            
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, current_gray, self.prev_pts, None)
        
        valid_dx, valid_dy = [], []
        if next_pts is not None:
            for i, (new, old) in enumerate(zip(next_pts, self.prev_pts)):
                if status[i] == 1:
                    valid_dx.append(new[0][0] - old[0][0])
                    valid_dy.append(new[0][1] - old[0][1])
                    
        if len(valid_dx) > 5:
            med_dx, med_dy = np.median(valid_dx), np.median(valid_dy)
            self.kf.statePost[2] = np.float32(med_dx)
            self.kf.statePost[3] = np.float32(med_dy)
            self.prev_pts = next_pts[status == 1].reshape(-1, 1, 2)
        else:
            self.is_tracking = False
            self.prev_pts = None
            return None, None, None
            
        self.prev_gray = current_gray.copy()
        prediction = self.kf.predict()
        pred_x, pred_y = int(prediction[0][0]), int(prediction[1][0])
        
        margin = int(self.last_box_size * 0.0 - self.last_box_size*0.2 ) 
        if (pred_x < -margin) or (pred_x > width + margin) or \
           (pred_y < -margin) or (pred_y > height + margin):
            self.is_tracking = False
            self.prev_pts = None
            return None, None, None
            
        return pred_x, pred_y, self.last_box_size
    
import math
import numpy as np

class OneEuroFilterVectorized:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0, freq=30):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.freq = freq
        
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x

        te = t - self.t_prev
        if te <= 0.0:
            return x # Prevent division by zero on duplicate timestamps

        ad = self.smoothing_factor(te, self.d_cutoff)
        dx = (x - self.x_prev) / te
        dx_hat = self.exponential_smoothing(ad, dx, self.dx_prev)

        # Dynamic cutoff: speed up filter when moving fast
        speed = np.linalg.norm(dx_hat, axis=-1, keepdims=True)
        cutoff = self.min_cutoff + self.beta * speed
        a = self.smoothing_factor(te, cutoff)
        
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

    def smoothing_factor(self, te, cutoff):
        r = 2 * math.pi * cutoff * te
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev


app = modal.App("kosha-sota-hamer-pipeline")

# ==========================================
# CLOUD VOLUMES 
# ==========================================
mano_volume = modal.Volume.from_name("mano-model-weights", create_if_missing=True)
yolo_volume = modal.Volume.from_name("yolo-master-volume", create_if_missing=True)

# Added Ultralytics and dependencies to the vision image
vision_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0", "libgles2", "libegl1", "libgl1-mesa-glx")
    .pip_install("ultralytics", "opencv-python-headless", "numpy", "torch", "torchvision","mediapipe")
)

hamer_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "build-essential","libegl1-mesa", "libegl1","libxrender1")
    .env({"PYOPENGL_PLATFORM": "egl","PYGLET_HEADLESS": "1"})
    .run_commands("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117")
    .run_commands("git clone --recursive https://github.com/geopavlakos/hamer.git /hamer")
    .run_commands(
        "cd /hamer && pip install -e .[all]",
        "cd /hamer && pip install -v -e third-party/ViTPose"
    )
)

# ==========================================
# 2. THE SOTA GPU REGRESSOR (HaMeR)
# ==========================================
@app.function(
    image=hamer_image, 
    gpu="A100", 
    volumes={"/mano_data": mano_volume}, 
    timeout=3600
)
def run_hamer_sota(image_crops_batch, is_right_hand_batch):
    import sys
    sys.path.append('/hamer') 
    import torch
    import numpy as np
    import os
    from hamer.models import load_hamer
    from torchvision import transforms

    device = torch.device('cuda')
    print(f"[A100 GPU] Booting HaMeR Regressor for {len(image_crops_batch)} crops...")
    
    if not os.path.exists('_DATA'):
        os.makedirs('_DATA', exist_ok=True)
    if not os.path.exists('_DATA/data'):
        os.symlink('/mano_data/data', '_DATA/data')
        
    checkpoint_path = "/mano_data/hamer_ckpts/checkpoints/hamer.ckpt"
    model, model_cfg = load_hamer(checkpoint_path)
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    refined_3d_batch = []

    with torch.no_grad():
        for crop, is_right in zip(image_crops_batch, is_right_hand_batch):
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                refined_3d_batch.append({"3d": [], "2d": []}) 
                continue
                
            input_tensor = transform(crop).unsqueeze(0).to(device)
            
            # The Left-Hand Flip Trick
            if not is_right:
                input_tensor = torch.flip(input_tensor, dims=[3])

            batch_dict = {'img': input_tensor}
            out = model(batch_dict)
            
            joints_3d = out['pred_keypoints_3d'][0].cpu().numpy()
            joints_2d = out['pred_keypoints_2d'][0].cpu().numpy()
            
            # The Clean Un-Flip
            if not is_right:
                joints_3d[:, 0] *= -1.0
                joints_2d[:, 0] *= -1.0
                
            refined_3d_batch.append({"3d": joints_3d.tolist(), "2d": joints_2d.tolist()})

    print("[A100 GPU] SOTA processing complete.")
    return refined_3d_batch


# ==========================================
# 3. THE CPU ORCHESTRATOR (YOLO Amodal)
# ========================
# ==========================================
# 3. THE CPU ORCHESTRATOR (YOLO -> MP Tasks -> HaMeR)
# ==========================================
# ==========================================
# 3. THE CPU ORCHESTRATOR (YOLO -> MP Tasks -> HaMeR)
# ==========================================
@app.function(
    image=vision_image, 
    cpu=4.0, 
    timeout=3600, 
    volumes={"/data": yolo_volume} 
)
def process_video_pipeline(video_bytes: bytes, width: int, height: int, fps: float):
    import cv2
    import numpy as np
    from ultralytics import YOLO
    import tempfile
    import urllib.request
    
    # Modern MediaPipe Tasks API
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_video:
        temp_video.write(video_bytes)
        cap = cv2.VideoCapture(temp_video.name)
        
        # Load YOLO
        yolo_path = "/data/best_hand_v4.pt" 
        print(f"[CPU Node] Loading Custom YOLO Model from {yolo_path}...")
        model = YOLO(yolo_path)
        
        # ---------------------------------------------------------
        # INITIALIZE MEDIAPIPE TASKS API (Using  urllib method)
        # ---------------------------------------------------------
        print("[CPU Node] Downloading and Initializing MediaPipe Tasks API...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", 
            "hand_landmarker.task"
        )
        
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1, # Strictly 1 because we feed it individual YOLO crops
            min_hand_detection_confidence=0.5
        )
        hands_detector = vision.HandLandmarker.create_from_options(options)
        
        final_frames_dict = {}
        batch_crops = []
        batch_handedness = []
        frame_metadata = []
        frame_id = 0
        # Initialize filters (Tune beta if it lags, tune min_cutoff if it jitter
        hand_filters = {
            "target-left": {
                "3d": OneEuroFilterVectorized(min_cutoff=0.9, beta=0.05, freq=fps),
                "2d": OneEuroFilterVectorized(min_cutoff=0.9, beta=0.05, freq=fps)
            },
            "target-right": {
                "3d": OneEuroFilterVectorized(min_cutoff=0.9, beta=0.05, freq=fps),
                "2d": OneEuroFilterVectorized(min_cutoff=0.9, beta=0.05, freq=fps)
            }
        }
        
        print("[CPU Node] Starting hybrid YOLO -> MediaPipe -> HaMeR extraction...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp = frame_id * (1.0 / fps)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 1. Run YOLO Inference (Amodal region isolation)
            results = model.predict(frame, verbose=False, conf=0.5)[0]
            
            found_left, found_right = False, False
            best_left_conf, best_right_conf = 0.0, 0.0
            left_box, right_box = None, None
            
            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                if cls_id == 0 and conf > best_left_conf:
                    left_box = (x1, y1, x2, y2)
                    best_left_conf = conf
                    found_left = True
                elif cls_id == 1 and conf > best_right_conf:
                    right_box = (x1, y1, x2, y2)
                    best_right_conf = conf
                    found_right = True

            valid_yolo_hands = []
            if found_left:
                valid_yolo_hands.append({"is_right": False, "bbox": left_box})
            if found_right:
                valid_yolo_hands.append({"is_right": True, "bbox": right_box})

            for hand_data in valid_yolo_hands:
                is_right = hand_data["is_right"]
                x1_yolo, y1_yolo, x2_yolo, y2_yolo = map(int, hand_data["bbox"])
                w = x2_yolo - x1_yolo
                h = y2_yolo - y1_yolo
                yolo_center_x = x1_yolo + w / 2.0
                yolo_center_y = y1_yolo + h / 2.0
                
                # =========================================================
                # Create a Padded Perfect Square for MediaPipe
                # MediaPipe goes blind if the crop is too tight or non-square.
                # =========================================================
                mp_size = int(max(w, h) * 1.3) # 30% padding for context
                mp_x1 = int(yolo_center_x - mp_size / 2)
                mp_y1 = int(yolo_center_y - mp_size / 2)
                mp_x2 = mp_x1 + mp_size
                mp_y2 = mp_y1 + mp_size
                
                # Safely slice the image using YOLO's coordinates
                # Safely extract and pad the crop with black bars if it's off-screen
                vx1, vy1 = max(0, mp_x1), max(0, mp_y1)
                vx2, vy2 = min(width, mp_x2), min(height, mp_y2)
                yolo_crop = rgb_frame[vy1:vy2, vx1:vx2]
                pad_left = vx1 - mp_x1
                pad_right = mp_x2 - vx2
                pad_top = vy1 - mp_y1
                pad_bottom = mp_y2 - vy2
                
                mp_ready_crop = np.pad(
                    yolo_crop, 
                    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                    mode='constant', constant_values=0
                )
                if mp_ready_crop.size == 0:
                    continue
                    
                # ---------------------------------------------------------
                # 2. RUN MEDIAPIPE TASKS ON THE YOLO CROP
                # ---------------------------------------------------------
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mp_ready_crop)
                detection_result = hands_detector.detect(mp_image)
                
                # If MediaPipe successfully finds the hand inside the YOLO crop
                if len(detection_result.hand_landmarks) > 0:
                    hand_landmarks = detection_result.hand_landmarks[0]
                    crop_h, crop_w, _ = mp_ready_crop.shape
                    
                    # Map MediaPipe's crop-relative coordinates back to global pixels
                    global_x_coords = [(lm.x * crop_w) + mp_x1 for lm in hand_landmarks]
                    global_y_coords = [(lm.y * crop_h) + mp_y1 for lm in hand_landmarks]
                    
                    x_min_mp, x_max_mp = min(global_x_coords), max(global_x_coords)
                    y_min_mp, y_max_mp = min(global_y_coords), max(global_y_coords)
                    center_x = (x_min_mp + x_max_mp) / 2.0
                    center_y = (y_min_mp + y_max_mp) / 2.0
                    box_size = int(max(x_max_mp - x_min_mp, y_max_mp - y_min_mp) * 1.8)
                else:
                    # FALLBACK: MediaPipe failed. Use YOLO amodal center
                    center_x = yolo_center_x
                    center_y = yolo_center_y
                    box_size = int(max(w, h) * 1.5)
                    
                    # ---------------------------------------------------------
                # 4. BUILD THE FINAL HAMER SQUARE 
                # ---------------------------------------------------------
                x1 = int(center_x - box_size / 2)
                y1 = int(center_y - box_size / 2)
                x2 = x1 + box_size
                y2 = y1 + box_size
                
                valid_x1, valid_y1 = max(0, x1), max(0, y1)
                valid_x2, valid_y2 = min(width, x2), min(height, y2)
                
                hand_crop_for_hamer = rgb_frame[valid_y1:valid_y2, valid_x1:valid_x2]
                
                pad_left = valid_x1 - x1
                pad_right = x2 - valid_x2
                pad_top = valid_y1 - y1
                pad_bottom = y2 - valid_y2
                
                perfect_square_crop = np.pad(
                    hand_crop_for_hamer, 
                    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                    mode='constant', constant_values=0
                )
                
                batch_crops.append(perfect_square_crop)
                batch_handedness.append(is_right)
                frame_metadata.append({
                    "frame_id": frame_id, 
                    "timestamp": timestamp,
                    "label": "target-right" if is_right else "target-left",
                    "bbox": [x1, y1, box_size, box_size]
                })
            # ---------------------------------------------------------
            # BATCH TRIGGER FOR HAMER
            # ---------------------------------------------------------
            if len(batch_crops) >= 60:
                refined_3d_batch = run_hamer_sota.remote(batch_crops, batch_handedness)
                update_frames_dictionary(final_frames_dict, frame_metadata, refined_3d_batch, width, height)
                batch_crops, batch_handedness, frame_metadata = [], [], []
                
            frame_id += 1
                
        # Flush any remaining frames at the end of the video
        if batch_crops:
            refined_3d_batch = run_hamer_sota.remote(batch_crops, batch_handedness)
            update_frames_dictionary(final_frames_dict, frame_metadata, refined_3d_batch, width, height)
            
        cap.release()
        return [final_frames_dict[fid] for fid in sorted(final_frames_dict.keys())]



'''def update_frames_dictionary(frames_dict, meta_list, refined_3d_list, width, height, hand_filters):
    for meta, kps in zip(meta_list, refined_3d_list):
        fid = meta["frame_id"]
        t = meta["timestamp"]
        label = meta["label"]  # "target-left" or "target-right"
        
        if fid not in frames_dict:
            frames_dict[fid] = {
                "frame_id": fid,
                "timestamp": round(t, 3),
                "camera_id": "ego_cam_front",
                "image_size": [width, height],
                "calibration_version": "v1.0",
                "target-left": {"status": "unknown", "reason_flags": ["missing hand"], "keypoints": []},
                "target-right": {"status": "unknown", "reason_flags": ["missing hand"], "keypoints": []}
            }
        
        # 1. Convert raw lists to NumPy arrays for vectorized filtering
        raw_3d = np.array(kps["3d"])
        raw_2d = np.array(kps["2d"])
        
        # 2. Apply the 1 Euro Filter (Bypass if the array is empty or safety-net zeros)
        if raw_3d.size > 0 and np.any(raw_3d):
            smooth_3d = hand_filters[label]["3d"](raw_3d, t)
            smooth_2d = hand_filters[label]["2d"](raw_2d, t)
        else:
            smooth_3d = raw_3d
            smooth_2d = raw_2d
            
        # 3. Format back into JSON lists
        # : Wrapped in float() to scrub NumPy data types and prevent json.dump crashes!
        formatted_3d = [
            {"id": i, "x": round(float(kp[0]), 4), "y": round(float(kp[1]), 4), "z": round(float(kp[2]), 4)} 
            for i, kp in enumerate(smooth_3d)
        ]
        
        formatted_2d = [
            {"id": i, "x": round(float(kp[0]), 4), "y": round(float(kp[1]), 4)} 
            for i, kp in enumerate(smooth_2d)
        ]
        
        frames_dict[fid][label] = {
            "status": "refined", 
            "reason_flags": [], 
            "bbox": meta["bbox"],
            "keypoints_3d": formatted_3d,
            "keypoints_2d": formatted_2d
        }'''
    
def update_frames_dictionary(frames_dict, meta_list, refined_3d_list, width, height):
    for meta, kps in zip(meta_list, refined_3d_list):
        fid = meta["frame_id"]
        
        if fid not in frames_dict:
            frames_dict[fid] = {
                "frame_id": fid,
                "timestamp": round(meta["timestamp"], 3),
                "camera_id": "ego_cam_front",
                "image_size": [width, height],
                "calibration_version": "v1.0",
                "target-left": {"status": "unknown", "reason_flags": ["missing hand"], "keypoints": []},
                "target-right": {"status": "unknown", "reason_flags": ["missing hand"], "keypoints": []}
            }
        
        formatted_3d = [{"id": i, "x": round(kp[0], 4), "y": round(kp[1], 4), "z": round(kp[2], 4)} for i, kp in enumerate(kps["3d"])]
        formatted_2d = [{"id": i, "x": round(kp[0], 4), "y": round(kp[1], 4)} for i, kp in enumerate(kps["2d"])]
        
        frames_dict[fid][meta["label"]] = {
            "status": "refined", 
            "reason_flags": [], 
            "bbox": meta["bbox"],
            "keypoints_3d": formatted_3d,
            "keypoints_2d": formatted_2d
        }

# ==========================================
# 4. LOCAL ENTRYPOINT
# ==========================================
@app.local_entrypoint()
def main(video_path: str, output_json: str):
    import cv2
    import os
    
    if not os.path.exists(video_path):
        print(f"Error: Could not find video at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    print(f"Reading {video_path}...")
    with open(video_path, "rb") as f:
        video_bytes = f.read()
        
    print("Initiating SOTA Modal Pipeline...")
    final_data = process_video_pipeline.remote(video_bytes, width, height, fps)
    
    with open(output_json, "w") as f:
        json.dump(final_data, f, indent=2)
        
    print(f"Success! Final canonical data saved to {output_json}")