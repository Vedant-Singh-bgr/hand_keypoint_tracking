def update_frames_dictionary(frames_dict, meta_list, refined_3d_list, width, height):
    """Merges hand predictions directly into matching frame blocks to handle multi-hand frames correctly"""
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