import io

import torch
from PIL import Image
from torchvision import models


class CatClassifier:
    # ImageNet indices corresponding to various cat classes:
    # 281: tabby, 282: tiger cat, 283: Persian cat, 284: Siamese cat, 285: Egyptian cat,
    # 286: cougar, 287: lynx, 288: leopard, 289: snow leopard, 290: jaguar, 291: lion, 292: tiger, 293: cheetah
    CAT_INDICES = set(range(281, 294))

    def __init__(self):
        # Load weights once at startup
        self.weights = models.MobileNet_V3_Small_Weights.DEFAULT
        self.model = models.mobilenet_v3_small(weights=self.weights)
        self.model.eval()
        self.transforms = self.weights.transforms()

    @torch.inference_mode()
    def predict_cat_probability(self, image_bytes: bytes) -> dict:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid image data: {str(e)}")

        tensor = self.transforms(image).unsqueeze(0)
        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

        # Aggregate probability across all feline/cat classes
        cat_prob = sum(probabilities[idx].item() for idx in self.CAT_INDICES)

        # Get top class name for context/debugging
        top_idx = torch.argmax(probabilities).item()
        top_label = self.weights.meta["categories"][top_idx]
        top_prob = probabilities[top_idx.conjugate()].item()

        return {
            "cat_probability": round(float(cat_prob), 4),
            "metadata": {
                "label": top_label,
                "confidence": round(float(top_prob), 4),
            },
        }
