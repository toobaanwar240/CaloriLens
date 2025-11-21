# classification/classifier.py

import torch
import numpy as np
from typing import Dict
from torchvision import transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

from .class_labels import FOOD_LABELS


class ConvNeXtClassifier:
    """
    Single-model food classifier based on ConvNeXt Base.
    """

    def __init__(self, weights_path: str = None):
        """
        weights_path: path to fine-tuned food classification weights (optional)
        """

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            # Load pretrained backbone
            self.model = convnext_base(
                weights=ConvNeXt_Base_Weights.IMAGENET1K_V1
            ).to(self.device)

            # Load custom head (optional)
            if weights_path:
                state = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state)

            self.model.eval()

            # Preprocessing
            w = ConvNeXt_Base_Weights.IMAGENET1K_V1
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=w.transforms.mean, std=w.transforms.std)
            ])

        except Exception as e:
            print("ConvNeXt failed to initialize:", e)
            self.model = None

    def predict(self, image: np.ndarray) -> Dict:
        """
        image: cropped BGR/NumPy food segment
        returns: { 'label': str, 'confidence': float }
        """

        if self.model is None:
            raise RuntimeError("ConvNeXt model is not available.")

        x = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]

        idx = int(torch.argmax(probs))
        return {
            "label": FOOD_LABELS[idx],
            "confidence": float(probs[idx])
        }
