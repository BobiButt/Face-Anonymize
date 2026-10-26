import cv2
import os
import config
from downloader import get_model
from detector import anonymize_frame

def main():
    model = get_model(config.MODEL_DIR, config.MODEL_NAME)
    
    if config.INPUT_TYPE == "image":
        os.makedirs(os.path.dirname(config.OUTPUT_PATH) or ".", exist_ok=True)
        input_path = config.INPUT_PATH
        # Resolve image path: try as-is, then under inputs/
        if not os.path.exists(input_path):
            alt = os.path.join("inputs", os.path.basename(input_path))
            if os.path.exists(alt):
                input_path = alt
        if not os.path.exists(input_path):
            print(f"[!] Image not found: {config.INPUT_PATH}")
            print(f"    Put your image in inputs/ and set INPUT_PATH in config.py")
            return
        img = cv2.imread(input_path)
        if img is None:
            print(f"[!] Could not read image: {input_path}")
            return
        result = anonymize_frame(img, model, config.BLUR_STRENGTH)
        out_path = config.OUTPUT_PATH.replace('.mp4', '.jpg')
        ok = cv2.imwrite(out_path, result)
        if ok:
            print(f"[+] Saved to {out_path}")
        else:
            print(f"[!] Failed to write image to {out_path}")
        
    elif config.INPUT_TYPE == "video":
        os.makedirs(os.path.dirname(config.OUTPUT_PATH) or ".", exist_ok=True)
        cap = cv2.VideoCapture(config.INPUT_PATH)
        out = cv2.VideoWriter(config.OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                               (int(cap.get(3)), int(cap.get(4))))
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = anonymize_frame(frame, model, config.BLUR_STRENGTH)
            out.write(frame)
        cap.release()
        out.release()
        
    elif config.INPUT_TYPE == "webcam":
        cap = cv2.VideoCapture(config.INPUT_PATH)
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = anonymize_frame(frame, model, config.BLUR_STRENGTH)
            cv2.imshow("Face Anonymizer - Press Q to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            # Window closed via X button
            if cv2.getWindowProperty("Face Anonymizer - Press Q to quit", cv2.WND_PROP_VISIBLE) < 1:
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
