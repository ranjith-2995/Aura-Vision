# type: ignore  # noqa: PGH003
"""
train_currency.py – Train an Indian Currency Classifier using MobileNetV2.

Dataset layout expected (already present in Datasets/):
    Datasets/
        fifty_new/      <- folder name becomes the class label
        fifty_old/
        five_hundred/
        hundred_new/
        hundred_old/
        ten_new/
        ten_old/
        twenty_new/
        twenty_old/
        two_hundred/
        two_thousand/

Outputs:
    currency_model.pt   - saved model weights
    currency_labels.txt - one label per line, matching class index order

Run:
    python train_currency.py
"""

import os
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
from torch.utils.data import DataLoader, random_split  # type: ignore
from torchvision import datasets, transforms, models  # type: ignore

# ─── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Datasets")
MODEL_OUT    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "currency_model.pt")
LABELS_OUT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "currency_labels.txt")
IMG_SIZE     = 224
BATCH_SIZE   = 32
EPOCHS       = 15
LR           = 1e-3
VAL_SPLIT    = 0.2   # 20% validation

# ─── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[TRAIN] Using device: {device}")
if device.type == "cuda":
    print(f"[TRAIN] GPU: {torch.cuda.get_device_name(0)}")

# ─── Transforms ────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet mean
                         [0.229, 0.224, 0.225]),   # ImageNet std
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─── Dataset ───────────────────────────────────────────────────────────────────
print(f"[TRAIN] Loading dataset from: {DATASET_DIR}")
full_dataset = datasets.ImageFolder(DATASET_DIR, transform=train_transform)

# Class names sorted alphabetically (same order as ImageFolder)
class_names = full_dataset.classes
num_classes = len(class_names)
print(f"[TRAIN] Found {num_classes} classes: {class_names}")
print(f"[TRAIN] Total images: {len(full_dataset)}")

# Train/val split
val_size   = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size],
                                 generator=torch.Generator().manual_seed(42))

# Override transform for val set
val_ds.dataset = datasets.ImageFolder(DATASET_DIR, transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=(device.type == "cuda"))
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=(device.type == "cuda"))

print(f"[TRAIN] Train: {train_size} | Val: {val_size}")

# ─── Model: MobileNetV2 (pretrained) ───────────────────────────────────────────
print("[TRAIN] Loading MobileNetV2 pretrained on ImageNet...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# Freeze all layers except the classifier
for param in model.features.parameters():
    param.requires_grad = False

# Replace the classifier head with our currency head
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.last_channel, num_classes),
)
model = model.to(device)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[TRAIN] Total params: {total_params:,}  Trainable: {trainable_params:,}")

# ─── Loss + Optimizer ──────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ─── Training Loop ─────────────────────────────────────────────────────────────
best_val_acc = 0.0

print(f"[TRAIN] Starting Phase 1: Training classifier only...")
for epoch in range(1, EPOCHS + 1):
    # ---- Train ----
    model.train()
    train_loss: float = 0.0
    train_correct: int = 0
    train_total: int = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss    += float(loss.item()) * int(images.size(0))
        preds          = outputs.argmax(dim=1)
        train_correct += int(torch.eq(preds, labels).sum().item())
        train_total   += int(images.size(0))

    scheduler.step()

    # ---- Validate ----
    model.eval()
    val_correct: int = 0
    val_total: int = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds   = outputs.argmax(dim=1)
            val_correct += int(torch.eq(preds, labels).sum().item())
            val_total   += int(images.size(0))

    train_acc = train_correct / train_total * 100
    val_acc   = val_correct   / val_total   * 100
    avg_loss  = train_loss    / train_total

    print(f"[Epoch {epoch:02d}/{EPOCHS}]  loss: {avg_loss:.4f}  "
          f"train_acc: {train_acc:.1f}%  val_acc: {val_acc:.1f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names":      class_names,
            "num_classes":      num_classes,
        }, MODEL_OUT)
        print(f"           ✓ Best model saved! (val_acc: {val_acc:.1f}%)")

# ─── Stage 2: Fine-Tuning ──────────────────────────────────────────────────────
print("\n[TRAIN] Starting Phase 2: Fine-tuning entire model (Unfreezing base)...")
# Unfreeze base layers
for param in model.features.parameters():
    param.requires_grad = True

# Lower learning rate for fine-tuning to avoid destroying weights
optimizer = optim.Adam(model.parameters(), lr=1e-5)

FINE_TUNE_EPOCHS = 10
for epoch in range(EPOCHS + 1, EPOCHS + FINE_TUNE_EPOCHS + 1):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss    += float(loss.item()) * int(images.size(0))
        train_correct += int(torch.eq(outputs.argmax(1), labels).sum().item())
        train_total   += int(images.size(0))

    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_correct += int(torch.eq(outputs.argmax(1), labels).sum().item())
            val_total   += int(images.size(0))

    val_acc = val_correct / val_total * 100
    print(f"[Epoch {epoch:02d}/{EPOCHS + FINE_TUNE_EPOCHS}] Fine-tune loss: {train_loss/train_total:.4f} val_acc: {val_acc:.1f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names":      class_names,
            "num_classes":      num_classes,
        }, MODEL_OUT)
        print(f"           ✓ Best model saved! (val_acc: {val_acc:.1f}%)")

# ─── Save Labels ───────────────────────────────────────────────────────────────
with open(LABELS_OUT, "w") as f:
    for name in class_names:
        f.write(name + "\n")

print(f"\n[TRAIN] ✅ Done! Best val accuracy: {best_val_acc:.1f}%")
print(f"[TRAIN] Model saved to: {MODEL_OUT}")
print(f"[TRAIN] Labels saved to: {LABELS_OUT}")
print("[TRAIN] Now run: python main.py")
