import os
import cv2
import tempfile
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

print('TensorFlow version:', tf.__version__)
print('Keras version:', tf.keras.__version__)

app = FastAPI(title="Deepfake Detection API")

MODEL_PATH = "xception_gru_model.keras"
NUM_FRAMES = 10
THRESHOLD = 0.55



# Load model when app starts
@app.on_event("startup")
def load_model():
    global model
    try:
        # model = tf.keras.models.load_model(MODEL_PATH)
        model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)

        print("✅ Model loaded successfully in backend!")
    except Exception as e:
        print("❌ Model loading failed:", e)

def preprocess_video(video_path, num_frames=NUM_FRAMES):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, frame_count - 1, num_frames).astype(int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (299, 299))
        frame = frame / 255.0
        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    if len(frames) < num_frames:
        pad_len = num_frames - len(frames)
        frames = np.pad(frames, ((0, pad_len), (0, 0), (0, 0), (0, 0)), "constant")
    return np.expand_dims(frames, axis=0)

@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    # Save uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        input_data = preprocess_video(tmp_path)
        pred = float(model.predict(input_data, verbose=0)[0][0])
        label = "FAKE" if pred >= THRESHOLD else "REAL"
        conf = pred if pred >= THRESHOLD else 1 - pred
        return JSONResponse({
            "filename": file.filename,
            "prediction": label,
            "confidence": round(conf, 3)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)

@app.get("/")
def home():
    return {"message": "Deepfake Detection API is running!"}
