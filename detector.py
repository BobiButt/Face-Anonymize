import cv2

def anonymize_frame(frame, model, blur_strength=51):
    results = model(frame, verbose=False)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # crop face and blur it
            face = frame[y1:y2, x1:x2]
            if face.size > 0:
                face = cv2.GaussianBlur(face, (blur_strength, blur_strength), 30)
                frame[y1:y2, x1:x2] = face
    return frame