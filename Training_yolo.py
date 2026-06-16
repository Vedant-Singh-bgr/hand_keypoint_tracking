import modal
import os

# 1. Define the Container Environment
app_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1-mesa-glx",      # Explicitly provides libGL.so.1
        "libglib2.0-0",         # Provides glib helper binaries (required for cv2 image handling)
        "libsm6",               # Session Management support for X11 rendering engines
        "libxext6",             # X11 miscellaneous extensions library
        "libxrender1"           # X-Render extension library for spatial pixel mapping
    )
    .pip_install(
        "ultralytics",
        "huggingface_hub",
        "opencv-python-headless",
        "roboflow",
        "pyyaml"
    )
)

# 2. Attach the Persistent Cloud Drive
volume = modal.Volume.from_name("yolo-master-volume", create_if_missing=True)
MOUNT_DIR = "/data"

app = modal.App(name="yolo-a100-pipeline", image=app_image)

# ==========================================
# PHASE 1: DATA PREPARATION (Runs on Cheap CPU)
# ==========================================
@app.function(volumes={MOUNT_DIR: volume}, secrets=[modal.Secret.from_name("hf-roboflow-keys")], timeout=86400)
def prepare_data():
    import json
    import tarfile
    import shutil
    import cv2
    import numpy as np
    from huggingface_hub import hf_hub_download, login, HfApi
    from roboflow import Roboflow
    import yaml
    import glob

    print("--- INITIATING DATA PREP PIPELINE ---")
    
    # Securely fetch API keys from the Modal Secret Vault
    hf_token = os.environ["HF_TOKEN"]
    rf_token = os.environ["ROBOFLOW_KEY"]
    login(token=hf_token)
    
    # ---------------------------------------------------------
    # THE PATH FIX: Everything MUST point to MOUNT_DIR
    # ---------------------------------------------------------
    MASTER_DIR = f"{MOUNT_DIR}/master_dataset_v8"
    HOT3D_DIR = f"{MOUNT_DIR}/hand_dataset_v8"
    TEMP_DIR = f"{MOUNT_DIR}/temp_tar_v8"
    ROBOFLOW_DIR = f"{MOUNT_DIR}/Roboflow_Data_v8" # Explicitly defining where Roboflow downloads
    
    if os.path.exists(f"{MASTER_DIR}/data.yaml"):
        print("Master dataset already exists on Volume. Skipping download phase.")
        return

    # Create Cloud Directories
    for split in ['train', 'val']:
        os.makedirs(f"{MASTER_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{MASTER_DIR}/labels/{split}", exist_ok=True)
        
    os.makedirs(f"{HOT3D_DIR}/images/train", exist_ok=True)
    os.makedirs(f"{HOT3D_DIR}/labels/train", exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    NUM_CLIPS = 20
    TARGET_STREAM = "214-1"  
    CLASS_MAP = {"left": 0, "right": 1}

    # ==========================================
    # HOT3D EXTRACTION
    # ==========================================
    print("Connecting to Meta's HOT3D Repository...")
    api = HfApi()
    all_files = api.list_repo_files(repo_id="bop-benchmark/hot3d", repo_type="dataset")
    aria_tars = [f for f in all_files if f.startswith("train_aria/clip-") and f.endswith(".tar")]
    tars_to_process = aria_tars[:NUM_CLIPS]

    converted_count = 0
    for i, repo_filepath in enumerate(tars_to_process):
        clip_filename = repo_filepath.split('/')[-1]
        print(f"Processing {clip_filename} ({i+1}/{NUM_CLIPS})...")
        
        try:
            tar_path = hf_hub_download(
                repo_id="bop-benchmark/hot3d", 
                filename=repo_filepath, 
                repo_type="dataset",
                token=hf_token,
                local_dir=TEMP_DIR
            )
            
            with tarfile.open(tar_path, "r") as tar:
                members = tar.getmembers()
                frames = {}
                for m in members:
                    prefix = m.name.split('.')[0]
                    ext = m.name[len(prefix)+1:]
                    if prefix not in frames:
                        frames[prefix] = {}
                    frames[prefix][ext] = m
                    
                for prefix, files in frames.items():
                    if 'hands.json' in files and 'cameras.json' in files:
                        hands_data = json.load(tar.extractfile(files['hands.json']))
                        
                        img_ext = f"image_{TARGET_STREAM}.jpg"
                        
                        # Check if the image exists before doing ANY math
                        if img_ext in files:
                            
                            # ---------------------------------------------------------
                            # Decode the image FIRST to get real pixel dimensions
                            # ---------------------------------------------------------
                            img_data = tar.extractfile(files[img_ext]).read()
                            np_arr = np.frombuffer(img_data, np.uint8)
                            img_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                            real_h, real_w, _ = img_cv2.shape
                            
                            # Grab the indisputable ground-truth dimensions from OpenCV
                            real_h, real_w, _ = img_cv2.shape
                            
                            yolo_lines = []
                            
                            for hand_type, class_id in CLASS_MAP.items():
                                if hand_type in hands_data and 'boxes_amodal' in hands_data[hand_type]:
                                    boxes = hands_data[hand_type]['boxes_amodal']
                                    if TARGET_STREAM in boxes:
                                        box = boxes[TARGET_STREAM]
                                        
                                        # ---------------------------------------------------------
                                        
                                        # ---------------------------------------------------------
                                        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
                                        true_w = xmax - xmin
                                        true_h = ymax - ymin
                                        
                                        # Calculate Center Points in absolute pixels
                                        x_center = xmin + (true_w / 2.0)
                                        y_center = ymin + (true_h / 2.0)
                                        
                                        x_norm = max(0.0, min(1.0, x_center / real_w))
                                        y_norm = max(0.0, min(1.0, y_center / real_h))
                                        w_norm = max(0.0, min(1.0, true_w / real_w))
                                        h_norm = max(0.0, min(1.0, true_h / real_h))
                                        
                                        # ---------------------------------------------------------
                                        
                                        # ---------------------------------------------------------
                                       
                                        yolo_lines.append(f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
                                        
                            if yolo_lines:
                                base_name = f"{clip_filename.split('.')[0]}_{prefix}"
                                rotated_yolo_lines = []
                                
                                for line in yolo_lines:
                                    class_id, x_c, y_c, w_n, h_n = map(float, line.split())
                                    
                                    # 90-Degree Clockwise Rotation Math
                                    new_x_c = max(0.0, min(1.0, 1.0 - y_c))
                                    new_y_c = max(0.0, min(1.0, x_c))
                                    
                                    # Note: Width becomes Height, Height becomes Width
                                    rotated_yolo_lines.append(f"{int(class_id)} {new_x_c:.6f} {new_y_c:.6f} {h_n:.6f} {w_n:.6f}")
                                
                                # ---------------------------------------------------------
                                

                                # Save Label
                                with open(f"{HOT3D_DIR}/labels/train/{base_name}.txt", 'w') as f:
                                    f.write("\n".join(rotated_yolo_lines))
                                    
                                # Rotate and Save Image
                                img_rotated = cv2.rotate(img_cv2, cv2.ROTATE_90_CLOCKWISE)
                                cv2.imwrite(f"{HOT3D_DIR}/images/train/{base_name}.jpg", img_rotated)
                                
                                converted_count += 1
            os.remove(tar_path)
        except Exception as e:
            print(f"  -> Skipping {clip_filename}: {e}")

    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    print(f"Extracted {converted_count} pristine HOT3D images.")

    # ==========================================
    # ROBOFLOW DOWNLOAD (Targeted to Volume)
    # ==========================================
    print("Downloading Roboflow Custom Dataset...")
    rf = Roboflow(api_key=rf_token)
    project = rf.workspace("vedants-workspace-dsekg").project("hand_detection-usrno")
    version = project.version(1)
    #Tell Roboflow exactly where to save it on the cloud drive
    dataset = version.download("yolov11", location=ROBOFLOW_DIR)
    ACTUAL_ROBO_PATH = dataset.location 
    print(f"Roboflow data found at: {ACTUAL_ROBO_PATH}")

    # ==========================================
    # SHUFFLE & MERGE PIPELINE
    # ==========================================
    import random
    print("Merging datasets...")
    hot3d_images = [f for f in os.listdir(f"{HOT3D_DIR}/images/train") if f.endswith('.jpg')]
    random.seed(42)
    random.shuffle(hot3d_images)

    val_split_index = int(len(hot3d_images) * 0.15)
    hot3d_val = hot3d_images[:val_split_index]
    hot3d_train = hot3d_images[val_split_index:]

    def move_files(file_list, source_base, dest_split):
        for f in file_list:
            base_name = os.path.splitext(f)[0]
            shutil.move(f"{source_base}/images/train/{f}", f"{MASTER_DIR}/images/{dest_split}/{f}")
            source_label_path = f"{source_base}/labels/train/{base_name}.txt"
            if os.path.exists(source_label_path):
                shutil.move(source_label_path, f"{MASTER_DIR}/labels/{dest_split}/{base_name}.txt")

    move_files(hot3d_train, HOT3D_DIR, 'train')
    move_files(hot3d_val, HOT3D_DIR, 'val')
    print("Scraping and Dynamically Splitting Custom Roboflow Data...")
    # Find every single jpg Roboflow downloaded, regardless of its original folder
    all_robo_images = glob.glob(f"{ACTUAL_ROBO_PATH}/*/images/*.jpg")
    random.shuffle(all_robo_images)

    robo_val_split_idx = int(len(all_robo_images) * 0.15)
    robo_val_imgs = all_robo_images[:robo_val_split_idx]
    robo_train_imgs = all_robo_images[robo_val_split_idx:]
    def process_robo_files(img_paths, dest_split):
        moved = 0
        for img_path in img_paths:
            filename = os.path.basename(img_path)
            base_name = os.path.splitext(filename)[0]
            
            # Predict the label path based on the image path structure
            lbl_path = img_path.replace("/images/", "/labels/").replace(".jpg", ".txt")
            
            shutil.move(img_path, f"{MASTER_DIR}/images/{dest_split}/{filename}")
            if os.path.exists(lbl_path):
                shutil.move(lbl_path, f"{MASTER_DIR}/labels/{dest_split}/{base_name}.txt")
            moved += 1
        return moved
    val_moved = process_robo_files(robo_val_imgs, 'val')
    train_moved = process_robo_files(robo_train_imgs, 'train')
    
    print(f"✅ Roboflow Split Complete: {train_moved} to Train | {val_moved} to Validation")

    

    # Generate the Data YAML
    yaml_content = {
        'train': f"{MASTER_DIR}/images/train",
        'val': f"{MASTER_DIR}/images/val",
        'names': {0: 'Left', 1: 'Right'}
    }
    with open(f"{MASTER_DIR}/data.yaml", 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print("Data compilation complete. Committing to Volume...")
    volume.commit() 


# ==========================================
# PHASE 2: TRAINING (Requests the A100 GPU)
# ==========================================
@app.function(
    gpu="A100", 
    volumes={MOUNT_DIR: volume}, 
    timeout=86400 
)
def train_yolo():
    from ultralytics import YOLO
    print("--- INITIATING A100 TRAINING PHASE ---")

    checkpoint_path = f"{MOUNT_DIR}/best_hand_v3.pt"
    
    # We use resume=False here because using best.pt as a starting point 
    
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}. Transfer Learning initiated...")
        model = YOLO(checkpoint_path)
    else:
        print("No checkpoint found. Starting fresh with YOLOv11m...")
        model = YOLO("yolo11m.pt")

    '''results = model.train(
        data=f"{MOUNT_DIR}/master_dataset_v8/data.yaml", 
        epochs=50, 
        imgsz=640,
        batch=64,
        workers=8, 
        patience=20,
        device=0, 
        optimizer="auto",
        project=f"{MOUNT_DIR}/yolo_outputs", 
        name="a100_tuning_run",
        resume=False 
    )'''
    metrics = model.val(data=f"{MOUNT_DIR}/master_dataset_v8/data.yaml")#for inference
    print(metrics.box.map) # Prints the mAP50-95

# 3. To run predictions and save output images/videos 
    

    volume.commit()
    print(f"Training Complete! Weights saved securely to {MOUNT_DIR}/yolo_outputs.")

# ==========================================
# EXECUTION ENTRY POINT
# ==========================================
@app.local_entrypoint()
def main():
    print("Deploying Pipeline to Modal...")
    prepare_data.remote()  
    train_yolo.remote()