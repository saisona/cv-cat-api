import io
import mimetypes
import random
from pathlib import Path
from typing import List, Tuple

from locust import FastHttpUser, LoadTestShape, constant_throughput, task
from PIL import Image

REAL_IMAGE_PATHS = [
    Path("tests/assets/cat_1.jpg"),
    Path("tests/assets/cat_2.webp"),
]

PayloadItem = Tuple[str, bytes, str]  # (filename, raw_bytes, content_type)
PAYLOAD_POOL: List[PayloadItem] = []


def _generate_synthetic_image_bytes(color: Tuple[int, int, int]) -> bytes:
    img = Image.new("RGB", (224, 224), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _initialize_payloads():
    # Load real images if present, fallback to placeholder
    for path in REAL_IMAGE_PATHS:
        content_type, _ = mimetypes.guess_type(str(path))
        content_type = content_type or "image/jpeg"

        if path.is_file():
            PAYLOAD_POOL.append((path.name, path.read_bytes(), content_type))
        else:
            PAYLOAD_POOL.append(
                (
                    path.name,
                    _generate_synthetic_image_bytes((200, 100, 50)),
                    "image/jpeg",
                )
            )

    # Add synthetic variation
    PAYLOAD_POOL.append(
        (
            "synth_cat.jpg",
            _generate_synthetic_image_bytes((128, 128, 128)),
            "image/jpeg",
        )
    )


_initialize_payloads()


# -------------------------------------------------------------------------
# 2. VIRTUAL USER DEFINITION (Fast gevent-based client)
# -------------------------------------------------------------------------
class MLBatchingUser(FastHttpUser):
    # Lock each user to ~20 requests/sec max to avoid local thread starving
    wait_time = constant_throughput(30)

    @task
    def test_predict_endpoint(self):
        filename, img_bytes, mime_type = random.choice(PAYLOAD_POOL)

        files = {"file": (filename, img_bytes, mime_type)}

        with self.client.post(
            "/predict",
            files=files,
            name="/predict",
            catch_response=True,
        ) as response:
            # 1. HTTP Status check
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text}")
                return

            # 2. Functional ML validation
            try:
                data = response.json()
                prob = data.get("probability")
                if prob is None or not (0.0 <= prob <= 1.0):
                    response.failure(f"Invalid ML payload response: {data}")
                else:
                    response.success()
            except Exception as e:
                response.failure(f"Malformed JSON: {str(e)}")


# -------------------------------------------------------------------------
# 3. REPRODUCIBLE STEP-RAMPING SHAPE (Find exact saturation point)
# -------------------------------------------------------------------------
class SteppedLoadShape(LoadTestShape):
    """
    Step-ramp test:
    - 00s -> 30s: 10 users  (Warmup / low traffic -> small batch sizes)
    - 30s -> 60s: 30 users  (Medium traffic -> dynamic batching kicks in)
    - 60s -> 90s: 60 users  (Saturation -> max batch size 16 fully saturated)
    - 90s -> 120s: 100 users (Stress / Queue backlog builds up)
    """

    stages = [
        {"duration": 30, "users": 10, "spawn_rate": 5},
        {"duration": 60, "users": 30, "spawn_rate": 10},
        {"duration": 90, "users": 60, "spawn_rate": 15},
        {"duration": 120, "users": 100, "spawn_rate": 20},
    ]

    def tick(self):
        run_time = self.get_run_time()

        for stage in self.stages:
            if run_time < stage["duration"]:
                return stage["users"], stage["spawn_rate"]

        return None  # Stop test after final stage
