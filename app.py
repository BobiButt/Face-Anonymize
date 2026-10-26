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
    img = cv2.imread(input_path)
    result = anonymize_frame(img, model, config.BLUR_STRENGTH)
    output_path = os.path.join("outputs", "anonymized_" + file.filename)
    cv2.imwrite(output_path, result)
    return templates.TemplateResponse("index.html", {"request": request, "result": output_path})

@app.get("/webcam")
def webcam():
    return {"status": "webcam endpoint - use main.py with INPUT_TYPE=webcam locally"}

@app.get("/outputs/{filename}")
def get_output(filename: str):
    return FileResponse(os.path.join("outputs", filename))