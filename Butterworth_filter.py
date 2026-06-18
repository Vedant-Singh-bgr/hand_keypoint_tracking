import json
import numpy as np
from scipy.signal import butter, filtfilt

def zero_phase_smooth(data_array, fps, cutoff_freq=3.0, order=2):
    """Zero-lag Butterworth filter for offline kinematics."""
    if len(data_array) < 9:
        return data_array # Bypass if sequence is too short to safely filter
        
    nyquist = 0.5 * fps
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    padlen = min(len(data_array) - 1, 15)
    return filtfilt(b, a, data_array, padlen=padlen)

def process_offline_json(input_json_path, output_json_path, fps=30.0):
    print(f"Loading {input_json_path} for offline 2D/3D smoothing...")
    with open(input_json_path, 'r') as f:
        frames = json.load(f)

    # ---------------------------------------------------------
    # 1. EXTRACT RAW TRAJECTORIES (2D and 3D)
    # ---------------------------------------------------------
    traj_3d = {"target-right": {}, "target-left": {}}
    traj_2d = {"target-right": {}, "target-left": {}}
    
    for frame in frames:
        for target in ["target-right", "target-left"]:
            if target in frame and frame[target]["status"] == "refined":
                kps_3d = frame[target]["keypoints_3d"]
                kps_2d = frame[target]["keypoints_2d"]
                
                # Initialize dictionaries on the first frame
                if not traj_3d[target]:
                    for kp in kps_3d:
                        traj_3d[target][kp["id"]] = {"x": [], "y": [], "z": []}
                    for kp in kps_2d:
                        traj_2d[target][kp["id"]] = {"x": [], "y": []}
                
                # Append 3D coordinates
                for kp in kps_3d:
                    jid = kp["id"]
                    traj_3d[target][jid]["x"].append(kp["x"])
                    traj_3d[target][jid]["y"].append(kp["y"])
                    traj_3d[target][jid]["z"].append(kp["z"])
                    
                # Append 2D coordinates
                for kp in kps_2d:
                    jid = kp["id"]
                    traj_2d[target][jid]["x"].append(kp["x"])
                    traj_2d[target][jid]["y"].append(kp["y"])

    # ---------------------------------------------------------
    # 2. APPLY ZERO-PHASE FILTER TO ALL AXES
    # ---------------------------------------------------------
    print("Applying Zero-Phase Butterworth Filter to 2D and 3D arrays...")
    smooth_3d = {"target-right": {}, "target-left": {}}
    smooth_2d = {"target-right": {}, "target-left": {}}
    
    for target in traj_3d:
        if not traj_3d[target]: continue
        
        # Smooth 3D
        for jid in traj_3d[target]:
            smooth_3d[target][jid] = {
                "x": zero_phase_smooth(np.array(traj_3d[target][jid]["x"]), fps),
                "y": zero_phase_smooth(np.array(traj_3d[target][jid]["y"]), fps),
                "z": zero_phase_smooth(np.array(traj_3d[target][jid]["z"]), fps)
            }
            
        # Smooth 2D
        for jid in traj_2d[target]:
            smooth_2d[target][jid] = {
                "x": zero_phase_smooth(np.array(traj_2d[target][jid]["x"]), fps),
                "y": zero_phase_smooth(np.array(traj_2d[target][jid]["y"]), fps)
            }

    # ---------------------------------------------------------
    # 3. REBUILD AND INJECT BACK INTO JSON
    # ---------------------------------------------------------
    print("Rebuilding JSON...")
    time_idx = {"target-right": 0, "target-left": 0}
    
    for frame in frames:
        for target in ["target-right", "target-left"]:
            if target in frame and frame[target]["status"] == "refined":
                idx = time_idx[target]
                
                # Overwrite the 3D keypoints
                for kp in frame[target]["keypoints_3d"]:
                    jid = kp["id"]
                    kp["x"] = round(float(smooth_3d[target][jid]["x"][idx]), 4)
                    kp["y"] = round(float(smooth_3d[target][jid]["y"][idx]), 4)
                    kp["z"] = round(float(smooth_3d[target][jid]["z"][idx]), 4)
                    
                # Overwrite the 2D keypoints
                for kp in frame[target]["keypoints_2d"]:
                    jid = kp["id"]
                    kp["x"] = round(float(smooth_2d[target][jid]["x"][idx]), 4)
                    kp["y"] = round(float(smooth_2d[target][jid]["y"][idx]), 4)
                
                time_idx[target] += 1

    with open(output_json_path, 'w') as f:
        json.dump(frames, f, indent=2)
        
    print(f"Success! Filtered 2D and 3D data saved to {output_json_path}")

# Example execution:
process_offline_json("hotel_midyolo_v3.json", "hotel_midyolo_smoothed.json", fps=30.0)