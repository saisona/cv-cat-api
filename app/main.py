import asyncio
import io
import logging
import sys
from contextlib import asynccontextmanager

import torchvision.models as models
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from PIL import Image

from batcher import DynamicBatcher

batcher: DynamicBatcher | None = None

# Preprocessing transforms directly from official weights
weights = models.MobileNet_V3_Small_Weights.DEFAULT
preprocess_transform = weights.transforms()

# Configure root logger so output is visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


def process_raw_image(data: bytes):
    try:
        with Image.open(io.BytesIO(data)) as img:
            img = img.convert("RGB")
            return preprocess_transform(img)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Corrupted or unsupported image file: {str(exc)}",
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    global batcher
    logger.info("Loading pre-trained MobileNetV3...")

    # Initialize base model
    model = models.mobilenet_v3_small(weights=weights)
    # Instantiate dynamic batcher (16 batch size, 10ms window)
    batcher = DynamicBatcher(model=model, max_batch_size=16, max_wait_time_ms=10.0)
    batcher.start()
    yield
    await batcher.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
def health():
    if batcher is None:
        raise HTTPException(status_code=503)
    return "ok"


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    logger.info("received a request")
    if batcher is None:
        raise HTTPException(status_code=500, detail="Batcher is not available")
    elif not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file format")

    # 1. Read & decode image bytes
    contents = await file.read()

    # Non-blocking image decode & tensor transform
    tensor = await asyncio.to_thread(process_raw_image, contents)

    # 3. Hand off to batcher
    prob = await batcher.predict(tensor)

    return {
        "probability": round(prob, 4),
    }
