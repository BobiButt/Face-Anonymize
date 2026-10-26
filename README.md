# Face Anonymize

A Python project that automatically detects and blurs (anonymizes) faces in images, videos, and webcam feeds using YOLOv8-face and OpenCV.

## What this project does

- Detects faces in an image, video, or webcam stream using a YOLOv8-face model.
- Blurs each detected face with a Gaussian blur so identities are hidden.
- Saves the anonymized result to an output file.
- Includes a FastAPI web UI to upload an image or video and download the anonymized result.

## Features

- Image anonymization
- Video anonymization
- Webcam anonymization (press `q` to quit)
- Auto-downloads the `yolov8n-face.pt` model if missing
- Configurable input type, paths, and blur strength via `config.py`
- FastAPI web interface with file upload

## Project structure

```
Face-Anonymize/
├── main.py           # CLI entry point (image / video / webcam)
├── app.py            # FastAPI web application
├── config.py         # Configuration (input type, paths, blur strength)
├── detector.py       # Face detection and blurring logic
├── downloader.py     # YOLO model download and loading
├── requirements.txt  # Python dependencies
├── templates/
│   └── index.html    # Web UI
├── models/           # Downloaded YOLO model (created at runtime)
├── inputs/           # Uploaded files (created at runtime)
└── outputs/          # Anonymized results (created at runtime)
```

## Installation

### CPU-only setup (recommended if you have no NVIDIA GPU)

ultralytics depends on PyTorch, and pip's default PyTorch includes NVIDIA/CUDA packages. To avoid that, install the CPU build first:

```bash
# Clone the repository
git clone https://github.com/BOBI-BUTT/Face-Anonymize.git
cd Face-Anonymize

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# 1. Install CPU-only PyTorch (no NVIDIA/CUDA packages)
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# 2. Install ultralytics without its default torch dependency (already installed)
pip install ultralytics --no-deps

# 3. Install the remaining requirements
pip install -r requirements.txt
```

### Standard setup (includes CUDA/NVIDIA packages)

```bash
git clone https://github.com/BOBI-BUTT/Face-Anonymize.git
cd Face-Anonymize
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Edit `config.py` to choose the input type and adjust settings:

```python
# Options: "image", "video", "webcam"
INPUT_TYPE = "webcam"

# If image/video, give path. If webcam, leave as 0
INPUT_PATH = 0

OUTPUT_PATH = "outputs/output.mp4"
MODEL_DIR = "models"
MODEL_NAME = "yolov8n-face.pt"  # face model - boxes cover faces only
BLUR_STRENGTH = 51              # must be odd for Gaussian blur
```

### Quick run (CPU-only)

```bash
cd /home/bobi/Documents/GitHub/Face-Anonymize
python -m venv venv
source venv/bin/activate        # Linux/Mac  (venv\Scripts\activate on Windows)
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install ultralytics --no-deps
pip install -r requirements.txt
python main.py
```

> **Note:** `yolov8n-face.pt` is a face-specific model downloaded from HuggingFace, so the blur box covers only the face (not the whole person). In the webcam window, press `q` or click the window's X button to close and exit the program.

## Usage

### Image

```bash
# Put your image in the inputs folder
cp your_photo.jpg inputs/

# Edit config.py:
# INPUT_TYPE = "image"
# INPUT_PATH = "inputs/your_photo.jpg"

python main.py
```
Result is saved to `outputs/your_photo.jpg` (`.jpg` regardless of input extension).

### FastAPI web UI

```bash
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`, upload an image or video, and download the anonymized result.

## Requirements

- Python 3.8+
- fastapi, uvicorn
- opencv-python, numpy
- ultralytics
- python-multipart, jinja2

Install all with:

```bash
pip install -r requirements.txt
```

## Author

**BOBI BUTT**
