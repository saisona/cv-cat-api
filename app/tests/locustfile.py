import io
import random
from pathlib import Path

from locust import HttpUser, between, task
from PIL import Image

REAL_IMAGE_PATH = Path("tests/assets/cat_1.jpg")  # Update with your actual image path


def load_real_image_bytes(path: Path) -> bytes:
    """Fallback generator if the real file doesn't exist locally."""
    if path.is_file():
        return path.read_bytes()

    # Fallback placeholder so test doesn't crash if asset is missing
    img = Image.new("RGB", (224, 224), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def generate_synthetic_image_bytes() -> bytes:
    """Generate a synthetic solid RGB image."""
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class CVInferenceUser(HttpUser):
    # Simulated pacing between requests
    wait_time = between(0.01, 0.05)

    def on_start(self):
        """Cache both image payloads in memory per virtual user."""
        self.real_image_bytes = load_real_image_bytes(REAL_IMAGE_PATH)
        self.synth_image_bytes = generate_synthetic_image_bytes()

        # Define choices and distribution weight (e.g. 50/50 or 70/30)
        self.payload_options = [
            ("real_cat.jpg", self.real_image_bytes),
            ("synth_cat.jpg", self.synth_image_bytes),
        ]
        # 50% chance real, 50% chance synthetic
        self.weights = [0.5, 0.5]

    @task
    def predict(self):
        # random.choices is clean, fast, and avoids disk I/O
        filename, img_bytes = random.choices(
            self.payload_options, weights=self.weights, k=1
        )[0]

        files = {"file": (filename, img_bytes, "image/jpeg")}

        with self.client.post("/predict", files=files, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"[{response.status_code}] {response.text}")
