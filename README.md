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

```bash
# Clone the repository
git clone https://github.com/BOBI-BUTT/Face-Anonymize.git
cd Face-Anonymize

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
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
MODEL_NAME = "yolov8n-face.pt"  # auto-downloaded if missing
BLUR_STRENGTH = 51              # must be odd for Gaussian blur
```

## Usage

### Command line

```bash
python main.py
```

- **Image**: reads `INPUT_PATH`, writes a `.jpg` result.
- **Video**: reads `INPUT_PATH`, writes an `.mp4` result.
- **Webcam**: shows a live window; press `q` to quit.

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
