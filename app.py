from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
import shutil, config, os
from downloader import get_model
import cv2
from detector import anonymize_frame

app = FastAPI()
templates = Jinja2Templates(directory="templates")
model = get_model(config.MODEL_DIR, config.MODEL_NAME)

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process(request: Request, file: UploadFile = File(...)):
    os.makedirs("inputs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    input_path = os.path.join("inputs", file.filename)
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ext = os.path.splitext(file.filename)[1].lower()
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    if ext in image_exts:
        img = cv2.imread(input_path)
        if img is None:
            return templates.TemplateResponse("index.html", {"request": request, "error": "Could not read image file."})
        result = anonymize_frame(img, model, config.BLUR_STRENGTH)
        output_path = os.path.join("outputs", "anonymized_" + file.filename)
        cv2.imwrite(output_path, result)
    elif ext in video_exts:
        output_path = os.path.join("outputs", "anonymized_" + os.path.splitext(file.filename)[0] + ".mp4")
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps is None:
            fps = 30
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = anonymize_frame(frame, model, config.BLUR_STRENGTH)
            out.write(frame)
        cap.release()
        out.release()
    else:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Unsupported file type. Upload an image or video."})

    return templates.TemplateResponse("index.html", {"request": request, "result": output_path})

@app.get("/webcam")
def webcam():
    return {"status": "webcam endpoint - use main.py with INPUT_TYPE=webcam locally"}

@app.get("/outputs/{filename}")
def get_output(filename: str):
    return FileResponse(os.path.join("outputs", filename))