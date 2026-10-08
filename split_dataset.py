from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

# Original dataset
SOURCE = Path(r"D:\Skin_Classification\dataset\organized")

# New split dataset
OUTPUT = Path(r"D:\Skin_Classification\dataset\binary_split")

classes = ["Healthy", "Unhealthy"]

for class_name in classes:
    source_folder = SOURCE / class_name

    images = list(source_folder.glob("*.jpg"))

    print(f"{class_name}: {len(images)} images")

    # 80% training, 20% validation
    train_images, val_images = train_test_split(
        images,
        test_size=0.20,
        random_state=42
    )

    # Create folders
    train_folder = OUTPUT / "train" / class_name
    val_folder = OUTPUT / "val" / class_name

    train_folder.mkdir(parents=True, exist_ok=True)
    val_folder.mkdir(parents=True, exist_ok=True)

    # Copy training images
    for image in train_images:
        shutil.copy2(image, train_folder / image.name)

    # Copy validation images
    for image in val_images:
        shutil.copy2(image, val_folder / image.name)

    print(f"  Train: {len(train_images)}")
    print(f"  Validation: {len(val_images)}")

print("\nDataset split completed!")