import os
import urllib.request
from ultralytics import YOLO

# Official Ultralytics model hub names (these download automatically)
OFFICIAL_MODELS = {
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
    "yolov8n-face.pt", "yolov8s-face.pt", "yolov8m-face.pt", "yolov8l-face.pt", "yolov8x-face.pt",
}

# Fallback direct URLs for community face models
FACE_MODEL_URLS = [
    "https://huggingface.co/deepghs/yolo-face/resolve/main/yolov8n-face/model.pt",
    "https://github.com/mahmoudmabdelrahman/yolov8-face/releases/download/v1.0/yolov8n-face.pt",
]


def get_model(model_dir, model_name):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)

    # If model exists locally, just load it
    if os.path.exists(model_path):
        print(f"[+] Model found: {model_path}")
        return YOLO(model_path)

    # Try official Ultralytics hub download
    try:
        print(f"[-] Model not found locally. Trying Ultralytics hub: {model_name}...")
        model = YOLO(model_name)
        model.save(model_path)
        print(f"[+] Downloaded and saved to {model_path}")
        return model
    except Exception as e:
        print(f"[-] Hub download failed for {model_name}: {e}")

    # Fallback: direct download for face models
    if model_name.endswith("-face.pt"):
        for url in FACE_MODEL_URLS:
            try:
                print(f"[-] Trying direct download from {url}...")
                urllib.request.urlretrieve(url, model_path)
                print(f"[+] Downloaded and saved to {model_path}")
                return YOLO(model_path)
            except Exception as e2:
                print(f"[-] Direct download failed ({url}): {e2}")

    raise FileNotFoundError(
        f"Could not obtain model '{model_name}'. "
        f"Please download it manually and place it at: {model_path}"
    )
