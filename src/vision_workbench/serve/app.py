"""FastAPI inference microservice for edge deployment."""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io

app = FastAPI(
    title="Vision Workbench Inference",
    description="Lightweight vision inference microservice",
    version="0.1.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_VERSION: str = "0.1.0"
_model: Optional[object] = None


@app.get("/v1/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    return JSONResponse({"inference_count": 0, "avg_latency_ms": 0})


@app.post("/v1/detect")
async def detect(file: UploadFile = File(...)):
    """Run object detection on an uploaded image.

    Returns:
        JSON with bounding boxes, classes, and confidence scores.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        img_array = np.array(image.convert("RGB"))
        img_bgr = img_array[:, :, ::-1].copy()

        # Placeholder inference
        results = {
            "detections": [],
            "image_shape": list(img_bgr.shape),
            "model_version": MODEL_VERSION,
        }
        return JSONResponse(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/classify")
async def classify(file: UploadFile = File(...)):
    """Run image classification on an uploaded image."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        return JSONResponse({"predictions": [], "image_size": list(image.size)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
