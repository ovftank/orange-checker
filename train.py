import torch
import traceback
import random
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

if __name__ == "__main__":
    try:
        seed = 42
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {DEVICE}")

        train_transform = transforms.Compose(
            [
                transforms.Resize((75, 75)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize((75, 75)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        train_dataset = datasets.ImageFolder(
            "./dataset/train", transform=train_transform
        )
        val_dataset = datasets.ImageFolder("./dataset/val", transform=test_transform)
        test_dataset = datasets.ImageFolder("./dataset/test", transform=test_transform)

        print(
            f"Classes: {len(train_dataset.classes)} | Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}"
        )

        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=32, shuffle=False, num_workers=2
        )
        test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=False, num_workers=2
        )

        model = models.resnet18(weights="IMAGENET1K_V1")

        num_features = model.fc.in_features
        classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(num_features, len(train_dataset.classes))
        )
        setattr(model, "fc", classifier)

        model = model.to(DEVICE)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        use_amp = DEVICE == "cuda"
        scaler = GradScaler(device=DEVICE) if use_amp else None

        EPOCHS = 50
        best_val_loss = float("inf")
        best_val_accuracy = 0.0
        patience = 7
        patience_counter = 0

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0

            for imgs, labels in train_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()

                if use_amp and scaler is not None:
                    with autocast(device_type=DEVICE):
                        preds = model(imgs)
                        loss = loss_fn(preds, labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    preds = model(imgs)
                    loss = loss_fn(preds, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

                    if use_amp:
                        with autocast(device_type=DEVICE):
                            preds = model(imgs)
                            loss = loss_fn(preds, labels)
                    else:
                        preds = model(imgs)
                        loss = loss_fn(preds, labels)

                    val_loss += loss.item()

                    _, predicted = torch.max(preds.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total

            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Acc: {val_accuracy:.2f}% | LR: {current_lr:.6f}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_accuracy = val_accuracy
                patience_counter = 0

                checkpoint = {
                    "state_dict": model.state_dict(),
                    "num_classes": len(train_dataset.classes),
                    "class_names": train_dataset.classes,
                    "model_type": "resnet18",
                    "transform": {
                        "input_size": (75, 75),
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                    },
                }
                model_path = "/content/drive/MyDrive/model.pth"
                torch.save(checkpoint, model_path)
                print(
                    f"✓ Saved | Val Loss: {best_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%"
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stop @ epoch {epoch + 1} | Best val loss: {best_val_loss:.4f}"
                    )
                    break
                elif patience_counter > 0:
                    print(f"  → No improvement ({patience_counter}/{patience})")

        print("\n" + "=" * 50)
        print("Final Test Evaluation")
        print("=" * 50)

        checkpoint = torch.load("/content/drive/MyDrive/model.pth")
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        test_loss = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

                if use_amp:
                    with autocast(device_type=DEVICE):
                        preds = model(imgs)
                        loss = loss_fn(preds, labels)
                else:
                    preds = model(imgs)
                    loss = loss_fn(preds, labels)

                test_loss += loss.item()

                _, predicted = torch.max(preds.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = 100 * test_correct / test_total

        print(f"Test Loss: {avg_test_loss:.4f} | Test Acc: {test_accuracy:.2f}%")
        print(
            f"Best Val Loss: {best_val_loss:.4f} | Best Val Acc: {best_val_accuracy:.2f}%"
        )
        print("=" * 50)

    except Exception:
        traceback.print_exc()
