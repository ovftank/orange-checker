import base64
import os
from io import BytesIO
from typing import Optional

import torch
from PIL import Image
from torchvision import models, transforms
from torch import nn

_model: Optional[torch.nn.Module] = None
_classes: Optional[list[str]] = None
_transform: Optional[transforms.Compose] = None
_device: Optional[str] = None


def _load_model():
    global _model, _classes, _transform, _device
    if _model is not None:
        return

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = "./model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model.pth không tồn tại tại {os.path.abspath(model_path)}")

    checkpoint = torch.load(model_path, map_location=_device)

    _classes = checkpoint["class_names"]

    model = models.resnet18(weights="IMAGENET1K_V1")
    num_features = model.fc.in_features
    classifier = nn.Sequential(
        nn.Dropout(0.5), nn.Linear(num_features, checkpoint["num_classes"])
    )
    setattr(model, "fc", classifier)

    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(_device)
    model.eval()
    _model = model

    transform_info = checkpoint["transform"]
    input_size = transform_info["input_size"]
    mean = transform_info["mean"]
    std = transform_info["std"]

    _transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def label(data, file_name):
    folders = [
        "avion",
        "bateau",
        "café",
        "camion",
        "chat",
        "chaussure",
        "cheval",
        "chien",
        "citrouille",
        "fleur",
        "girafe",
        "lampe",
        "lion",
        "montgolfiere",
        "montre",
        "moto",
        "oiseau",
        "papillon",
        "piscine",
        "poisson",
        "théière",
        "tomate",
        "tortue",
        "tracteur",
        "violon",
        "éléphant",
    ]

    try:
        _load_model()
    except Exception as e:
        print(f"load model fail: {e}")
        return

    try:
        if _model is None or _transform is None or _classes is None or _device is None:
            raise RuntimeError("model chưa được load")

        img_data = base64.b64decode(data)
        img = Image.open(BytesIO(img_data)).convert("RGB")

        assert _transform is not None and _device is not None
        img_tensor = _transform(img).unsqueeze(0).to(_device)  # type: ignore

        with torch.no_grad():
            outputs = _model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            class_idx = predicted.item()

        assert _classes is not None
        if class_idx < len(_classes):
            label_result = _classes[class_idx]  # type: ignore
        else:
            print(f"class_idx {class_idx} out of range, skip")
            return

        if label_result in folders:
            with open(f"class/{label_result}/{file_name}", "wb") as f:
                f.write(img_data)
            print(f"class/{label_result}/{file_name}")
    except Exception as e:
        print(f"predict fail: {e}")
        return
