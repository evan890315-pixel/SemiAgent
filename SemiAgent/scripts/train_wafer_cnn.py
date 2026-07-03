"""
scripts/train_wafer_cnn.py

晶圓缺陷分類器訓練腳本
架構：ResNet18 Transfer Learning
輸入：WM-811K 晶圓圖片（已分好 train/val/test）
      圖片顏色編碼：紅=背景、綠=正常晶粒、藍=缺陷晶粒
輸出：models/wafer_classifier/best.pth

Label Mapping（9種 → 4種）：
  Scratch              → scratch
  Center / Loc / Random / Near-full → particle
  Edge-Ring / Donut / Edge-Loc      → void
  none                 → normal
"""

import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── 設定 ─────────────────────────────────────────────────────────
DATA_ROOT  = Path("data/wafer")
OUTPUT_DIR = Path("models/wafer_classifier")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE  = 32
NUM_EPOCHS  = 15
LR          = 1e-4
NUM_WORKERS = 0          # Windows 用 0
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# WM-811K 9種 → SemiAgent 4種 label mapping
LABEL_MAP = {
    "Scratch":   "scratch",
    "Center":    "particle",
    "Loc":       "particle",
    "Random":    "particle",
    "Near-full": "particle",
    "Edge-Ring": "void",
    "Donut":     "void",
    "Edge-Loc":  "void",
    "none":      "normal",
}

TARGET_CLASSES = ["normal", "particle", "scratch", "void"]


# ─── 自訂 Dataset ─────────────────────────────────────────────────
class WaferDataset(Dataset):
    def __init__(self, root_dir: str, split: str, transform=None):
        self.transform = transform
        self.samples   = []
        self.class2idx = {c: i for i, c in enumerate(TARGET_CLASSES)}

        split_dir = Path(root_dir) / split
        for original_label in os.listdir(split_dir):
            mapped_label = LABEL_MAP.get(original_label)
            if mapped_label is None or mapped_label not in self.class2idx:
                continue
            label_idx = self.class2idx[mapped_label]
            img_dir   = split_dir / original_label
            for img_file in img_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_file), label_idx))

        print(f"[{split}] 總共 {len(self.samples)} 張圖片")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # 保留 RGB 顏色資訊（紅/綠/藍各代表不同區域）
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((64, 64, 3), dtype=np.uint8)

        # OpenCV 預設 BGR，轉成 RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label


# ─── Transforms ───────────────────────────────────────────────────
# ImageNet normalize（和 ResNet18 預訓練一致）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ─── 主訓練流程 ───────────────────────────────────────────────────
def main():
    print(f"🖥️  Device: {DEVICE}")
    print(f"📂 資料路徑: {DATA_ROOT.absolute()}")
    print(f"🎨 前處理：RGB 顏色保留（紅=背景/綠=正常/藍=缺陷）")

    train_ds = WaferDataset(DATA_ROOT, "train", train_transform)
    val_ds   = WaferDataset(DATA_ROOT, "validation",   val_transform)
    test_ds  = WaferDataset(DATA_ROOT, "test",  val_transform)

    # 處理類別不平衡
    labels    = [s[1] for s in train_ds.samples]
    class_cnt = np.bincount(labels, minlength=len(TARGET_CLASSES))
    print(f"\n📊 訓練集類別分布：")
    for i, cls in enumerate(TARGET_CLASSES):
        print(f"   {cls}: {class_cnt[i]} 張")

    weights        = 1.0 / (class_cnt + 1e-6)
    sample_weights = torch.tensor([weights[l] for l in labels], dtype=torch.float)
    sampler        = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    # ResNet18 Transfer Learning
    print("\n 載入 ResNet18 預訓練權重...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # 凍結 layer1~3，只訓練 layer4 + FC
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(512, len(TARGET_CLASSES))
    )
    model = model.to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"   可訓練參數：{trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_acc = 0.0
    best_model   = None
    history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n 開始訓練（{NUM_EPOCHS} epochs）...\n")

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()

        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * imgs.size(0)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == lbls).sum().item()
            train_total   += imgs.size(0)

        scheduler.step()

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                outputs   = model(imgs)
                loss      = criterion(outputs, lbls)
                val_loss += loss.item() * imgs.size(0)
                preds     = outputs.argmax(dim=1)
                val_correct += (preds == lbls).sum().item()
                val_total   += imgs.size(0)

        t_loss  = train_loss / train_total
        t_acc   = train_correct / train_total
        v_loss  = val_loss / val_total
        v_acc   = val_correct / val_total
        elapsed = time.time() - t0

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}]  "
              f"Train Loss={t_loss:.4f} Acc={t_acc:.4f}  "
              f"Val Loss={v_loss:.4f} Acc={v_acc:.4f}  "
              f"({elapsed:.1f}s)")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_model   = copy.deepcopy(model.state_dict())
            print(f"   ✅ 新最佳 Val Acc：{best_val_acc:.4f}")

    # 儲存最佳模型
    model.load_state_dict(best_model)
    save_path = OUTPUT_DIR / "best.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes":          TARGET_CLASSES,
        "class2idx":        {c: i for i, c in enumerate(TARGET_CLASSES)},
        "val_acc":          best_val_acc,
        "label_map":        LABEL_MAP,
    }, save_path)
    print(f"\n✅ 最佳模型儲存至：{save_path}")
    print(f"   最佳 Val Acc：{best_val_acc:.4f}")

    # Test
    print("\n📊 Test Set 評估...")
    model.eval()
    test_correct, test_total = 0, 0
    class_correct = [0] * len(TARGET_CLASSES)
    class_total   = [0] * len(TARGET_CLASSES)

    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            outputs = model(imgs)
            preds   = outputs.argmax(dim=1)
            test_correct += (preds == lbls).sum().item()
            test_total   += imgs.size(0)
            for i in range(len(lbls)):
                l = lbls[i].item()
                class_correct[l] += (preds[i] == lbls[i]).item()
                class_total[l]   += 1

    print(f"\n🎯 Test Accuracy：{test_correct/test_total:.4f} ({test_correct}/{test_total})")
    print("\n各類別準確率：")
    for i, cls in enumerate(TARGET_CLASSES):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            print(f"   {cls:12s}：{acc:.4f} ({class_correct[i]}/{class_total[i]})")

    # 訓練曲線
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"],   label="Val")
    ax1.set_title("Loss"); ax1.legend(); ax1.set_xlabel("Epoch")
    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"],   label="Val")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curve.png")
    print(f"\n📈 訓練曲線儲存至：{OUTPUT_DIR}/training_curve.png")


if __name__ == "__main__":
    main()