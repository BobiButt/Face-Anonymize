import cv2
import config
from downloader import get_model
from detector import anonymize_frame

def main():
    model = get_model(config.MODEL_DIR, config.MODEL_NAME)
    
    if config.INPUT_TYPE == "image":
        img = cv2.imread(config.INPUT_PATH)
        result = anonymize_frame(img, model, config.BLUR_STRENGTH)
        cv2.imwrite(config.OUTPUT_PATH.replace('.mp4','.jpg'), result)
        print(f"Saved to {config.OUTPUT_PATH}")
        
    elif config.INPUT_TYPE == "video":
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
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
