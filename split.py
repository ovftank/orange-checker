import random
from pathlib import Path
from shutil import copy


def split_dataset(source_dir="class", train_ratio=0.7, val_ratio=0.15):
    source = Path(source_dir)
    dataset_dir = Path("dataset")
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    test_dir = dataset_dir / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    classes = [d for d in source.iterdir() if d.is_dir()]

    total_train = 0
    total_val = 0
    total_test = 0

    for class_dir in classes:
        class_name = class_dir.name
        files = list(class_dir.glob("*.jpeg"))

        if not files:
            print(f"skip {class_name}: no files")
            continue

        random.shuffle(files)
        train_idx = int(len(files) * train_ratio)
        val_idx = train_idx + int(len(files) * val_ratio)

        train_files = files[:train_idx]
        val_files = files[train_idx:val_idx]
        test_files = files[val_idx:]

        train_class_dir = train_dir / class_name
        val_class_dir = val_dir / class_name
        test_class_dir = test_dir / class_name
        train_class_dir.mkdir(exist_ok=True)
        val_class_dir.mkdir(exist_ok=True)
        test_class_dir.mkdir(exist_ok=True)

        for f in train_files:
            copy(f, train_class_dir / f.name)

        for f in val_files:
            copy(f, val_class_dir / f.name)

        for f in test_files:
            copy(f, test_class_dir / f.name)

        print(f"{class_name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
        total_train += len(train_files)
        total_val += len(val_files)
        total_test += len(test_files)

    print(f"\ntotal: {total_train} train, {total_val} val, {total_test} test")


if __name__ == "__main__":
    random.seed(42)
    split_dataset()
