import cv2
import numpy as np

def get_skin_mask(roi_bgr):
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    # Merge back and convert to BGR
    limg = cv2.merge((cl, a_channel, b_channel))
    clahe_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Convert to YCrCb
    ycrcb = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2YCrCb)
    
    # Standard human skin YCrCb strict bounds
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
        measurement = np.array([[np.float32(center_x)], [np.float32(center_y)]])
        self.kf.correct(measurement)
        
        self.last_box_size = box_size
        self.prev_gray = frame_gray.copy()
        
        full_mask = np.zeros_like(frame_gray)
        roi_bgr = frame_bgr[valid_y1:valid_y2, valid_x1:valid_x2]
        
        if roi_bgr.size > 0:
            skin_mask_roi = get_skin_mask(roi_bgr)
            full_mask[valid_y1:valid_y2, valid_x1:valid_x2] = skin_mask_roi
            self.prev_pts = cv2.goodFeaturesToTrack(frame_gray, maxCorners=50, qualityLevel=0.1, minDistance=5, mask=full_mask)
            self.is_tracking = True
        else:
            self.is_tracking = False
            
        self.kf.predict() 

    def predict_with_flow(self, current_gray, width, height):
        if not self.is_tracking or self.prev_pts is None:
            return None, None, None 
            
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, current_gray, self.prev_pts, None)
        
        valid_dx, valid_dy = [], []
        if next_pts is not None:
            for i, (new, old) in enumerate(zip(next_pts, self.prev_pts)):
                if status[i] == 1:
                    valid_dx.append(new[0][0] - old[0][0])
                    valid_dy.append(new[0][1] - old[0][1])
                    
        if len(valid_dx) > 5:   #tunable parameter to track hand
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