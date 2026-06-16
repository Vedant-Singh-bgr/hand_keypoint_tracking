# SOTA HaMeR Pipeline

A  Human Mesh Recovery (HMR) pipeline designed to extract robust 3D MANO hand kinematics from egocentric video in occluded/dim hotel environments. 


## 📁 Repository Structure

```text
keypoint_tracking/
├── main.py # Modal cloud orchestrator & API entrypoint
├── Training_yolo.py # Train yolo v11m model
├── src/
           
│   ├── tracking.py      # Skin-masking, Kalman, and Optical Flow physics
│   └── utils.py         # Canonical JSON formatting 
│   └── visualiser.py         # to test json on video
├── data/                # Local video storage (Ignored by Git)
├── output/              # Generated JSON kinematic files (Ignored by Git)
├── .gitignore
└── requirements.txt

