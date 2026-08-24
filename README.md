# Real-Time Computer Vision Inference Service (Dynamic Batching)

High-throughput, low-latency computer vision microservice designed to serve cat probability inference via FastAPI and PyTorch TorchScript. Engineered for high concurrency environments using an asynchronous dynamic queue-batching engine optimized for CPU and GPU edge/cloud deployments.

---

## 1. Architectural Highlights

* **Dynamic Request Batching (`batcher.py`)**:
  * Asynchronous queue-based aggregation using `asyncio.Queue` and thread pool execution (`asyncio.to_thread`).
  * Configurable latency budget (`max_wait_time_ms = 10 ms`) and capacity limit (`max_batch_size = 16`).
  * Per-request `asyncio.Future` promises ensure zero cross-talk between concurrent client requests.
* **Inference Engine Optimization**:
  * **Model**: MobileNetV3-Small (`torchvision.models.mobilenet_v3_small` with `DEFAULT` weights).
  * **TorchScript Freezing**: Traced and optimized using `torch.jit.trace` and `torch.jit.optimize_for_inference` for minimal Python runtime overhead.
  * **Hardware Agnostic**: Automatic device placement (`cuda` with FP16 precision if an NVIDIA GPU is available; falls back to CPU SIMD/AVX execution).
* **Asynchronous Web Layer (`main.py`)**:
  * FastAPI with Uvicorn ASGI runtime.
  * Lifespan-managed initialization and shutdown for resource safety.
  * Direct byte-stream decoding to PIL and tensor conversion (`torchvision.transforms`).

---

## 2. API Specification

### Health Check

```http
GET /health
```

Returns Ok if service is healthy and batcher instance is set returns an HTTP Response with 503 Status Code

### Predict

```http
POST /predict
Content-Type: multipart/form-data
```

#### Parameters

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `file` | `multipart/form-data` | Yes | Raw image file (JPEG, PNG, WebP) |

#### Sample Request

```bash
curl -X POST "<http://localhost:8080/predict>" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.jpg;type=image/jpeg"
```

returns an object containing the probability of a presence of a cat inside the given sample like

```json
{
  "probability": 0.9842
}
```
