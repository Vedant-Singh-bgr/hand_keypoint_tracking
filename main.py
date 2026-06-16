import modal
import json
import os
import time

# Local module imports


app = modal.App("kosha-sota-hamer-pipeline")

# ==========================================
# CLOUD VOLUMES & IMAGES
# ==========================================
mano_volume = modal.Volume.from_name("mano-model-weights", create_if_missing=True)
yolo_volume = modal.Volume.from_name("yolo-master-volume", create_if_missing=True)

vision_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0", "libgles2", "libegl1", "libgl1-mesa-glx")
    .pip_install("ultralytics", "opencv-python-headless", "numpy", "torch", "torchvision")
    .add_local_dir("src", remote_path="/root/src")
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
    .add_local_dir("src", remote_path="/root/src")
)

# ==========================================
# SOTA GPU REGRESSOR (HaMeR)
# ==========================================
@app.function(
    image=hamer_image, 
    gpu="A100", 
    volumes={"/mano_data": mano_volume},
    timeout=1800 
)
def run_hamer(image_crops_batch, is_right_hand_batch):
    import sys
    sys.path.append('/hamer') 
    import torch
    import numpy as np
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
    #Transform to resize cropped bboxes to Hamer compatible 256x256
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    refined_3d_batch = []
    #inference
    with torch.no_grad():
        for crop, is_right in zip(image_crops_batch, is_right_hand_batch):
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                refined_3d_batch.append({"3d": [], "2d": []}) 
                continue
                
            input_tensor = transform(crop).unsqueeze(0).to(device)
            
            if not is_right:
                input_tensor = torch.flip(input_tensor, dims=[3])

            batch_dict = {'img': input_tensor}
            out = model(batch_dict)
            
            joints_3d = out['pred_keypoints_3d'][0].cpu().numpy()
            joints_2d = out['pred_keypoints_2d'][0].cpu().numpy()
            
            if not is_right:
                joints_3d[:, 0] *= -1.0
                joints_2d[:, 0] *= -1.0
                
            refined_3d_batch.append({"3d": joints_3d.tolist(), "2d": joints_2d.tolist()})#extracting 2d/3d keypoints

    print("[A100 GPU] SOTA processing complete.")
    return refined_3d_batch

# ==========================================
# CPU ORCHESTRATOR (YOLO Amodal)
# ==========================================
@app.function(
    image=vision_image, 
    cpu=4.0, 
    timeout=1800, 
    volumes={"/data": yolo_volume}
)
def process_video_pipeline(video_bytes: bytes, width: int, height: int, fps: float):
    import cv2
    import numpy as np
    from ultralytics import YOLO
    from src.tracking import KalmanOpticalTracker
    from src.utils import update_frames_dictionary
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_video:
        temp_video.write(video_bytes)
        cap = cv2.VideoCapture(temp_video.name)
        
        yolo_path = "/data/best_hand.pt" #best weight file of YOLO on modal
        print(f"[CPU Node] Loading Custom YOLO Model from {yolo_path}...")
        model = YOLO(yolo_path)
        
        final_frames_dict = {}
        batch_crops, batch_handedness, frame_metadata = [], [], []
        frame_id = 0
        
        tracker_left = KalmanOpticalTracker()#initialise trackers to use kalman filter
        tracker_right = KalmanOpticalTracker()
        #inference
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp = frame_id * (1.0 / fps)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            
            results = model.predict(frame, verbose=False, conf=0.5)[0]
            
            found_left, found_right = False, False
            best_left_conf, best_right_conf = 0.0, 0.0
            left_box, right_box = None, None
            
            
            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                #left =0, right =1
                if cls_id == 0 and conf > best_left_conf:
                    left_box = (x1, y1, x2, y2)
                    best_left_conf = conf
                    found_left = True
                elif cls_id == 1 and conf > best_right_conf:
                    right_box = (x1, y1, x2, y2)
                    best_right_conf = conf
                    found_right = True

            valid_hands = []
            if found_left:
                valid_hands.append({"is_right": False, "bbox": left_box, "tracker": tracker_left})
            if found_right:
                valid_hands.append({"is_right": True, "bbox": right_box, "tracker": tracker_right})

            for hand_data in valid_hands:
                is_right = hand_data["is_right"]
                x1_raw, y1_raw, x2_raw, y2_raw = hand_data["bbox"]
                tracker = hand_data["tracker"]
                
                w = x2_raw - x1_raw
                h = y2_raw - y1_raw
                center_x = x1_raw + (w / 2.0)
                center_y = y1_raw + (h * 0.35)
                box_size = int(w * 1.2)
                
                tmp_x1 = max(0, int(center_x - box_size/2))
                tmp_y1 = max(0, int(center_y - box_size/2))
                tmp_x2 = min(width, int(center_x + box_size/2))
                tmp_y2 = min(height, int(center_y + box_size/2))
                
                #update tracker for ekf
                tracker.update_with_detector(
                    center_x, center_y, box_size, 
                    gray_frame, frame, 
                    tmp_x1, tmp_y1, tmp_x2, tmp_y2
                )
                
                x1 = int(center_x - box_size / 2)
                y1 = int(center_y - box_size / 2)
                x2, y2 = x1 + box_size, y1 + box_size
                valid_x1, valid_y1 = max(0, x1), max(0, y1)
                valid_x2, valid_y2 = min(width, x2), min(height, y2)
                #padding to ensure there is no distortion while resizing when bbox moves to hamer
                hand_crop = rgb_frame[valid_y1:valid_y2, valid_x1:valid_x2]
                pad_left, pad_right = valid_x1 - x1, x2 - valid_x2
                pad_top, pad_bottom = valid_y1 - y1, y2 - valid_y2
                
                perfect_square_crop = np.pad(
                    hand_crop, 
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

            # Optical Flow Fallbacks (only executed when either left/right is mising)
            if not found_left:
                left_pred = tracker_left.predict_with_flow(gray_frame, width, height)
                if left_pred[0] is not None:
                    pred_x, pred_y, box_size = left_pred
                    x1 = int(pred_x - box_size / 2)
                    y1 = int(pred_y - box_size / 2)
                    x2, y2 = x1 + box_size, y1 + box_size
                    valid_x1, valid_y1 = max(0, x1), max(0, y1)
                    valid_x2, valid_y2 = min(width, x2), min(height, y2)
                    
                    hand_crop = rgb_frame[valid_y1:valid_y2, valid_x1:valid_x2]
                    pad_left, pad_right = valid_x1 - x1, x2 - valid_x2
                    pad_top, pad_bottom = valid_y1 - y1, y2 - valid_y2
                    perfect_square_crop = np.pad(hand_crop, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
                    
                    
                    batch_crops.append(perfect_square_crop)
                    batch_handedness.append(False)
                    frame_metadata.append({"frame_id": frame_id, "timestamp": timestamp, "label": "target-left", "bbox": [x1, y1, box_size, box_size]})
                    
            if not found_right:
                right_pred = tracker_right.predict_with_flow(gray_frame, width, height)
                if right_pred[0] is not None:
                    pred_x, pred_y, box_size = right_pred
                    x1 = int(pred_x - box_size / 2)
                    y1 = int(pred_y - box_size / 2)
                    x2, y2 = x1 + box_size, y1 + box_size
                    valid_x1, valid_y1 = max(0, x1), max(0, y1)
                    valid_x2, valid_y2 = min(width, x2), min(height, y2)
                    
                    hand_crop = rgb_frame[valid_y1:valid_y2, valid_x1:valid_x2]
                    pad_left, pad_right = valid_x1 - x1, x2 - valid_x2
                    pad_top, pad_bottom = valid_y1 - y1, y2 - valid_y2
                    perfect_square_crop = np.pad(hand_crop, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
                    
                    batch_crops.append(perfect_square_crop)
                    batch_handedness.append(True)
                    frame_metadata.append({"frame_id": frame_id, "timestamp": timestamp, "label": "target-right", "bbox": [x1, y1, box_size, box_size]})
             # giving a100 gpu in batches of 60 crops at a timr   
            if len(batch_crops) >= 60:
                refined_3d_batch = run_hamer.remote(batch_crops, batch_handedness)
                update_frames_dictionary(final_frames_dict, frame_metadata, refined_3d_batch, width, height)
                batch_crops, batch_handedness, frame_metadata = [], [], []
            
            frame_id += 1
        
        #for remaining batch crops    
        if batch_crops:
            refined_3d_batch = run_hamer.remote(batch_crops, batch_handedness)
            update_frames_dictionary(final_frames_dict, frame_metadata, refined_3d_batch, width, height)
                
        cap.release()
        return [final_frames_dict[fid] for fid in sorted(final_frames_dict.keys())]

# ==========================================
# LOCAL ENTRYPOINT
# ==========================================
@app.local_entrypoint()
def main(video_path: str, output_json: str):
    import cv2
    
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