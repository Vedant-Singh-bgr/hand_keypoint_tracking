import cv2
import json

# The standard 21-joint MANO kinematic chain
BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring
    (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

def run_live_debugger(video_path, json_path):
    
    print("Loading JSON data...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # Map data by frame_id for instantaneous O(1) lookup
    frame_dict = {f["frame_id"]: f for f in data}
    
    # 2. Boot the video stream
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    print("=========================================")
    print("LIVE DEBUGGER CONTROLS:")
    print("[SPACEBAR] : Pause / Play")
    print("[N]        : Next Frame (while paused)")
    print("[Q]        : Quit")
    print("=========================================")

    frame_id = 0
    paused = False
    
    cv2.namedWindow("SOTA 2D Ground Truth", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream.")
                break
                
            #drawing tracked frame data
            if frame_id in frame_dict:
                frame_data = frame_dict[frame_id]
                
                # Colors: Crimson for Left, Emerald for Right
                for hand_key, color in [("target-left", (102, 51, 255)), ("target-right", (136, 255, 17))]:
                    hand = frame_data[hand_key]
                    
                    if hand["status"] == "refined":
                        x_min, y_min, box_w, box_h = hand["bbox"]
                        kps_2d = hand["keypoints_2d"]
                        raw_points = [{"id": kp["id"], "x": kp["x"], "y": kp["y"]} for kp in kps_2d]
                        
                        # 2. Dynamic Auto-Scaler (Your Spanning Logic)
                        x_vals = [p["x"] for p in raw_points]
                        y_vals = [p["y"] for p in raw_points]
                        net_center_x = (min(x_vals) + max(x_vals)) / 2.0
                        net_center_y = (min(y_vals) + max(y_vals)) / 2.0
                        
                        # Find the internal network span (the tiny decimal meters)
                        net_span = max(max(x_vals) - min(x_vals), max(y_vals) - min(y_vals))
                        if net_span == 0: net_span = 1.0

                        # 3. Scale to fit exactly 55% of the new 1.8x Perfect Square Box
                        target_pixel_span = box_w * 0.55
                        scale_factor = target_pixel_span / net_span

                        # 4. Map to OpenCV Pixels using the exact center
                        pixel_center_x = x_min + (box_w / 2.0)
                        pixel_center_y = y_min + (box_h / 2.0)

                        pixel_coords = []
                        for p in sorted(raw_points, key=lambda k: k["id"]):
                            global_x = int(pixel_center_x + (p["x"] - net_center_x) * scale_factor)
                            global_y = int(pixel_center_y + (p["y"] - net_center_y) * scale_factor)

                            pixel_coords.append((global_x, global_y))
                            
                            cv2.circle(frame, (global_x, global_y), 3, color, -1)
                        # Draw the bone connections
                        for j1, j2 in BONES:
                            cv2.line(frame, pixel_coords[j1], pixel_coords[j2], color, 1)
                            
                        # Draw the neural network's bounding box limit
                        cv2.rectangle(frame, (x_min, y_min), (x_min + box_w, y_min + box_h), color, 2)

        
        cv2.imshow("SOTA 2D Ground Truth", frame)

        # Keyboard Control Logic
        delay = 0 if paused else 30  
        key = cv2.waitKey(delay) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):  # Spacebar to toggle pause
            paused = not paused
        elif key == ord('n') and paused:  # Step one frame forward
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            continue # Bypass the frame_id increment below so we don't double count
            
        if not paused:
            frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ensure these filenames match your actual local files
    run_live_debugger("Testing_yolo.mp4", "hotel_YOLO_hamer_test7.json")