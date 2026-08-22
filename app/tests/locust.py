import os

from locust import HttpUser, between, task

# Read the test image into memory once at startup
IMAGE_PATH = "test_cat.jpg"
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"Please place a valid '{IMAGE_PATH}' in the current directory."
    )

with open(IMAGE_PATH, "rb") as f:
    IMAGE_BYTES = f.read()


class ImageInferenceUser(HttpUser):
    # Simulate constant back-to-back requests (high throughput test)
    wait_time = between(0.01, 0.05)

    @task
    def predict_image(self):
        files = [{"file": ("test_cat.jpg", IMAGE_BYTES, "image/jpeg")}]
        with self.client.post("/predict", files=files, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "probability" in data:
                    response.success()
                else:
                    response.failure("Malformed JSON response")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
