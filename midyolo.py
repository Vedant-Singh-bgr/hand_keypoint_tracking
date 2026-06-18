import modal
import json
import time

# ==========================================
# 1. DEFINE CLOUD INFRASTRUCTURE
# ==========================================
import numpy as np
import cv2

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
