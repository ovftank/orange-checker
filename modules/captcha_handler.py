
from io import BytesIO
import torch
from PIL import Image
from torch import Tensor, nn
from torchvision import models, transforms
import os
import threading

class CaptchaHandler:
    def __init__(self,model_path) -> None:
         self.device = "cuda" if torch.cuda.is_available() else "cpu"
         self.model_path = model_path
         self.class_names = None
         self._model_loaded = False
         self._model_lock = threading.Lock()
    def _load_model(self) -> None:
        # Check if model is already loaded
        if self._model_loaded:
            return

        with self._model_lock:
            # Double-check after acquiring lock
            if self._model_loaded:
                return

            if not os.path.exists(self.model_path):
                raise FileNotFoundError("model not found")

            checkpoint = torch.load(self.model_path, map_location=self.device)
            model = models.resnet18(weights=None)
            num_features = model.fc.in_features
            classifier = nn.Sequential(
                nn.Dropout(0.5), nn.Linear(num_features, checkpoint["num_classes"])
            )
            setattr(model, "fc", classifier)
            model.load_state_dict(checkpoint["state_dict"])
            model = model.to(self.device)
            model.eval()

            self.model = model
            self.class_names = checkpoint["class_names"]
            self.transform_config = checkpoint.get("transform", {})
            self._model_loaded = True

    def predict(self,img_bytes:bytes):
        self._load_model()
        mean = self.transform_config.get("mean", [0.485, 0.456, 0.406])
        std = self.transform_config.get("std", [0.229, 0.224, 0.225])
        input_size = self.transform_config.get("input_size", (75, 75))

        transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        img  =  Image.open(BytesIO(img_bytes)).convert("RGB")
        img_tensor = transform(img)
        assert isinstance(img_tensor, Tensor), "transform should return Tensor"
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_idx = torch.max(probs, dim=1)
        if self.class_names is not None:
            class_name = self.class_names[top_idx.item()]
            confidence = top_prob.item()
            print(class_name)
            print(confidence)
