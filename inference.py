import torch
from PIL import Image
from torch import nn, Tensor
from torchvision import models, transforms


def load_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    classifier = nn.Sequential(
        nn.Dropout(0.5), nn.Linear(num_features, checkpoint["num_classes"])
    )
    setattr(model, "fc", classifier)

    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint["class_names"], checkpoint.get("transform", {})


def preprocess_image(image_path, transform_config):
    mean = transform_config.get("mean", [0.485, 0.456, 0.406])
    std = transform_config.get("std", [0.229, 0.224, 0.225])
    input_size = transform_config.get("input_size", (75, 75))

    transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img)
    assert isinstance(img_tensor, Tensor), "transform should return Tensor"
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


def predict(model, image_tensor, class_names, device="cpu"):
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_idx = torch.max(probs, dim=1)

    class_name = class_names[top_idx.item()]
    confidence = top_prob.item()

    return class_name, confidence


if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) < 2:
        sys.exit(1)

    image_path = sys.argv[1]

    model_path = "model.pth"
    if not os.path.exists(model_path):
        print(f"error: k thay model: {model_path}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, class_names, transform_config = load_model(model_path, device)
    image_tensor = preprocess_image(image_path, transform_config)
    class_name, confidence = predict(model, image_tensor, class_names, device)

    print(class_name)
