import os
import torch
from threading import Lock
from PIL import Image
from torchvision import models, transforms
from torch import nn, Tensor
from config import Config

class ModelLoader:
    """Quản lý việc tải và xử lý mô hình ML"""
    
    def __init__(self, model_path: str = Config.MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.class_names = None
        self.transform_config = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model_lock = Lock()
    
    def load_model(self) -> None:
        """Tải mô hình từ file"""
        if self.model is not None:
            return

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        model = models.resnet18(weights=None)
        num_features = model.fc.in_features
        classifier = nn.Sequential(
            nn.Dropout(Config.MODEL_DROPOUT), 
            nn.Linear(num_features, checkpoint["num_classes"])
        )
        setattr(model, "fc", classifier)
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(self.device)
        model.eval()

        self.model = model
        self.class_names = checkpoint["class_names"]
        self.transform_config = checkpoint.get("transform", {})
        print(f"Model loaded on device: {self.device}")
        return self.model
    
    def preprocess_image(self, image_path: str) -> Tensor:
        """Tiền xử lý ảnh cho mô hình"""
        if not self.transform_config:
            raise RuntimeError("Model not loaded")
        
        transform = transforms.Compose([
            transforms.Resize(Config.MODEL_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=Config.MODEL_MEAN, 
                std=Config.MODEL_STD
            ),
        ])
        
        img = Image.open(image_path).convert("RGB")
        print("check8")
        img_tensor = transform(img)
        assert isinstance(img_tensor, Tensor), "Transform should return Tensor"
        return img_tensor.unsqueeze(0)
    
    def predict(self, image_tensor: Tensor) -> tuple[str, float]:
        """Dự đoán kết quả từ ảnh"""
        if not self.model or not self.class_names:
            raise RuntimeError("Model not loaded")

        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_idx = torch.max(probs, dim=1)

        class_name = self.class_names[top_idx.item()]
        confidence = top_prob.item()
        return class_name, confidence
