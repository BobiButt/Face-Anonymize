
# config.py - change this file to switch input

# Options: "image", "video", "webcam"
INPUT_TYPE = "webcam"  

# If image/video, give path. If webcam, leave as 0
INPUT_PATH = 0  
# Examples:
# INPUT_PATH = "inputs/test.jpg"
# INPUT_PATH = "inputs/test.mp4"
# INPUT_PATH = 0  # for webcam

OUTPUT_PATH = "outputs/output.mp4"
MODEL_DIR = "models"
MODEL_NAME = "yolov8n-face.pt" # face model - boxes cover faces only
BLUR_STRENGTH = 51