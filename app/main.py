from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile, status

from model import CatClassifier

model: CatClassifier | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model
    # Load model during app startup (avoids cold start penalty on first request)
    model = CatClassifier()
    yield
    # Clean up resources if necessary
    model = None


app = FastAPI(
    title="Computer Vision Classification API", version="1.0.0", lifespan=lifespan
)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """
    Healthz check performed by the Kubernetes cluster for both readiness/liveness of the application
    """
    return {"status": "healthy"}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if model == None:
        raise Exception("model is not loaded")
    elif file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG, PNG, or WebP.",
        )

    try:
        content = await file.read()
        results = model.predict(content)
        return {"filename": file.filename, **results}
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err)
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed.",
        )
