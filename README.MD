# DeepFake Detection API

This repository provides a small FastAPI service that loads an Xception-GRU model to detect whether an uploaded video is REAL or FAKE.

## Quick links
- Swagger UI / interactive docs (test endpoints in browser): http://127.0.0.1:8000/docs

## Project layout

- `app.py` — FastAPI application. Exposes `/` (root) and `/predict` (POST) endpoints.
- `model_tf.py` — preprocessing and model-loading / inference helpers. Uses lazy loading (`get_model()`) to avoid heavy work at import time.
- `saved_models/` — directory that contains the saved Keras model: `xception_gru_model.keras`.
- `requirements.txt` — Python dependencies (use a virtual environment).

## Requirements

- Python 3.10+ (this project was tested on CPython 3.11+ as well)
- A virtual environment is strongly recommended.

## Setup (Windows / PowerShell)

1. Create and activate a venv (from project root):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Upgrade pip and install requirements:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Start the API server (development):

```powershell
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/docs` in your browser to try the API and upload a video file via the Swagger UI.

## API Usage

1) Root (health)

GET `/` → Returns a simple JSON message confirming the service is running.

2) Predict (upload a video)

POST `/predict`
- Content type: `multipart/form-data`
- Field name: `file` (the uploaded video file, e.g. `.mp4`)

Response (JSON):
- `label`: `"REAL"` or `"FAKE"`
- `confidence`: float (0.00 - 1.00)

Example using PowerShell + curl.exe (PowerShell `curl` is an alias; use `curl.exe` to call real curl):

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" -F "file=@C:\path\to\video.mp4"
```

Python example (requests):

```python
import requests

url = "http://127.0.0.1:8000/predict"
with open(r"C:\path\to\video.mp4", "rb") as f:
		files = {"file": f}
		r = requests.post(url, files=files)
		print(r.status_code, r.json())
```

Browser/front-end example (Fetch + FormData):

```javascript
const fd = new FormData();
fd.append('file', fileInput.files[0]);
fetch('/predict', { method: 'POST', body: fd })
	.then(r => r.json())
	.then(console.log);
```

Important: If you open `/predict` in your browser directly you will send a GET request and receive HTTP 405 Method Not Allowed — the endpoint only accepts POST with a file upload.

## Model loading behavior

- The model file `saved_models/xception_gru_model.keras` is heavy; to avoid problems during process startup the code uses lazy loading:

	- `model_tf.get_model()` will load and cache the model on first use.
	- `predict_video()` calls `get_model()` internally.

- If you want to force-load the model manually (to see load-time errors immediately), run:

```powershell
python -c "from model_tf import get_model; get_model()"
```

## Known issues & troubleshooting

1) Model load errors (example: "Cannot convert '10' to a shape" inside TimeDistributed)

- Cause: In many cases this is a Keras/TensorFlow serialization/deserialization mismatch. The model might have been saved with a different TF/Keras version than the one you have locally.

- Your environment vs Colab example:
	- Colab: TF 2.19.0 | Keras 3.10.0
	- Local: TF 2.20.0 | Keras 3.12.0

- Fixes (ranked):
	1. Recreate the original environment (install TensorFlow 2.19.0 locally) and try loading. If successful, re-save the model in a more portable format (SavedModel) or re-save with your local TF version.
		 ```powershell
		 # inside venv
		 python -m pip install 'tensorflow==2.19.0'
		 python -c "from model_tf import get_model; get_model()"
		 ```
	2. Re-save model in Colab as TensorFlow SavedModel and copy that folder to `saved_models/`, then load that saved model locally. In Colab:
		 ```python
		 model = tf.keras.models.load_model('xception_gru_model.keras')
		 model.save('/content/xception_gru_savedmodel')  # SavedModel (directory)
		 ```
	3. If you cannot change local TF, reimplement the model architecture in code and load only weights (or export weights from Colab and import them here).

2) 405 Method Not Allowed when calling `/predict`

- Cause: You used GET on an endpoint that only accepts POST (file upload). Use POST with multipart/form-data.

3) Model too large or slow to load

- Lazy loading mitigates worker-spawn crashes. For production, consider loading the model once on startup in a single process (not multiple workers) or using a model-serving solution (TensorFlow Serving, TorchServe, Triton).

## Development notes

- Logging: Most errors are printed to stdout/stderr. When running under `uvicorn --reload` you will see logs in the terminal.
- If you change the model or TF versions, re-save models and re-test.

## If you want help debugging the `.keras` file

I can inspect the saved `.keras` file (it's a zip-like archive) and show the layer config that contains the offending value — this often reveals which layer serialized an integer where a tuple/shape is expected. If you want that, grant me permission to open the binary file or paste the load-time stack trace here and I'll parse it.

---

If you want, I can also add a tiny `health` GET handler for `/predict` to return a user-friendly message on GET (instead of 405), or add a separate `/load_model` endpoint to force-load the model and report status. Tell me which you'd prefer and I can add it.