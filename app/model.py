import io

import torch
from PIL import Image
from torchvision import models


class CatClassifier:
    # ImageNet indices corresponding to various cat classes:
    # 281: tabby, 282: tiger cat, 283: Persian cat, 284: Siamese cat, 285: Egyptian cat,
    CAT_INDICES = set(range(281, 285))

    def __init__(self):
        # Load weights once at startup
        self.weights = models.MobileNet_V3_Small_Weights.DEFAULT
        self.model = models.mobilenet_v3_small(weights=self.weights)
        self.model.eval()

        # Restrict PyTorch CPU threads to avoid CPU thrashing under high concurrency
        torch.set_num_threads(4)
        self.transforms = self.weights.transforms()

    @torch.inference_mode()
    def predict(self, image_bytes: bytes) -> dict:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid image data: {str(e)}")

        tensor = self.transforms(image).unsqueeze(0)
        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

        # Aggregate probability across all feline/cat classes
        cat_prob = sum(probabilities[idx].item() for idx in self.CAT_INDICES)

        return {
            "probability": round(float(cat_prob), 4),
        }
