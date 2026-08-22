import io
from contextlib import asynccontextmanager

import torchvision.models as models
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from batcher import DynamicBatcher

batcher: DynamicBatcher | None = None

# Lightweight image preprocessing
preprocess_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global batcher
    # Initialize base model
    model = models.mobilenet_v3_small(weights=None, num_classes=2)
    # Instantiate dynamic batcher (16 batch size, 10ms window)
    batcher = DynamicBatcher(model=model, max_batch_size=16, max_wait_time_ms=10.0)
    batcher.start()
    yield
    await batcher.stop()


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if batcher is None:
        raise HTTPException(status_code=500, detail="Batcher is not available")
    elif not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file format")

    # 1. Read & decode image bytes
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # 2. Preprocess to tensor: shape (3, 224, 224)
    tensor = preprocess_transform(image)

    # 3. Hand off to batcher
    prob = await batcher.predict(tensor)

    return {
        "probability": round(prob, 4),
    }
