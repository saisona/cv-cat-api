from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile, status

from repo.model import CatClassifier

model_service: CatClassifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_service
    # Load model during app startup (avoids cold start penalty on first request)
    model_service = CatClassifier()
    yield
    # Clean up resources if necessary
    model_service = None


app = FastAPI(
    title="Computer Vision Classification API", version="1.0.0", lifespan=lifespan
)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "healthy"}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG, PNG, or WebP.",
        )

    try:
        content = await file.read()
        results = model_service.predict_cat_probability(content)
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
