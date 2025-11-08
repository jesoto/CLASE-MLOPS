import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # quieter TF logs; set before importing TF

import re
import pickle
from typing import List, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json
import uvicorn

# ---------- Config ----------
MODEL_DIR = os.getenv("MODEL_DIR", "models")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pickle")
MODEL_JSON_PATH = os.path.join(MODEL_DIR, "model.json")
MODEL_WEIGHTS_PATH = os.path.join(MODEL_DIR, ".model.weights.h5")  # keep your path; change if needed
MAXLEN = 1000
LABELS: List[Literal["Detractor", "Pasivo", "Promotor"]] = ["Detractor", "Pasivo", "Promotor"]

app = FastAPI(title="NPS LSTM API", version="1.0.0")

# Will be populated at startup
app.state.tokenizer = None
app.state.model = None

# ---------- Esquema ----------
class PredictIn(BaseModel):
    comentario: str

class PredictOut(BaseModel):
    pred: Literal["Detractor", "Pasivo", "Promotor"]
    probs: List[float]  # model softmax outputs [p0, p1, p2]

# ---------- Helpers ----------
_clean_re = re.compile(r"[^A-Za-z0-9\s]")

def _preprocess(text: str) -> List[List[int]]:
    t = text.lower()
    t = _clean_re.sub("", t)
    seq = app.state.tokenizer.texts_to_sequences([t])
    return pad_sequences(seq, maxlen=MAXLEN, padding="pre")

def _predict_label(probs: np.ndarray) -> str:
    idx = int(np.argmax(probs))
    return LABELS[idx]

# ---------- Lifecycle ----------
@app.on_event("startup")
def load_artifacts():
    # Load tokenizer
    with open(TOKENIZER_PATH, "rb") as f:
        app.state.tokenizer = pickle.load(f)

    # Load model from JSON + weights
    with open(MODEL_JSON_PATH, "r") as f:
        model_json = f.read()
    model = model_from_json(model_json)
    model.load_weights(MODEL_WEIGHTS_PATH)

    # Optionally do a warm-up (tiny dummy input) so first request is fast
    _ = model.predict(pad_sequences([[0]], maxlen=MAXLEN, padding="pre"))

    app.state.model = model

# ---------- Endpoints ----------
@app.get("/health")
def health():
    ok = (app.state.model is not None) and (app.state.tokenizer is not None)
    return {
        "status": "ok" if ok else "not_ready",
        "model_loaded": app.state.model is not None,
        "tokenizer_loaded": app.state.tokenizer is not None,
        "maxlen": MAXLEN,
        "labels": LABELS,
    }

@app.post("/predict",
           response_model=PredictOut,
           summary="Predecir NPS a partir de un comentario",
           description="Devuelve la predicción de NPS (Detractor, Pasivo, Promotor) y las probabilidades asociadas."
           )
def predict(inp: PredictIn):
    if not inp.comentario or not inp.comentario.strip():
        raise HTTPException(status_code=400, detail="Comentario vacío.")

    X = _preprocess(inp.comentario)
    probs = app.state.model.predict(X)[0].tolist() 
    pred = _predict_label(np.array(probs))
    return PredictOut(pred=pred, probs=[float(p) for p in probs])

# uvicorn app_fastapi:app --host 0.0.0.0 --port 8000