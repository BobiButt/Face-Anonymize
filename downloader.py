import os
from ultralytics import YOLO

def get_model(model_dir, model_name):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    
    # If model exists, just load
    if os.path.exists(model_path):
        print(f"[+] Model found: {model_path}")
        return YOLO(model_path)
    
    # If not, download it first in specific folder
    print(f"[-] Model not found. Downloading {model_name} to {model_dir}...")
    model = YOLO(model_name) # downloads from ultralytics
    model.save(model_path) # save to our folder
    print(f"[+] Downloaded and saved to {model_path}")
    return model
