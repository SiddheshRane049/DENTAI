"""
╔══════════════════════════════════════════════════════════════════════╗
║  DentAI — Fine-Tuning: Add Filling & Crown to Existing Model        ║
║  Run this as a Kaggle Notebook (P100 GPU)                            ║
║  Step 1: Paste into a new Code cell. Step 2: Run All.                ║
╚══════════════════════════════════════════════════════════════════════╝

INSTRUCTIONS:
1. Create a NEW Kaggle Notebook.
2. Enable GPU: Settings → Accelerator → GPU P100.
3. Upload your existing best.pt as a Kaggle Dataset and add it to this notebook.
4. Paste this entire script into a single code cell and RUN.
5. After completion, download best_finetuned.pt from /kaggle/working/
6. Rename to dental_yolov8.pt and place in your Django models/ folder.
"""

# ─── 1. INSTALL DEPENDENCIES ───────────────────────────────────────────────────
import subprocess, sys

def pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

pip("ultralytics")
pip("roboflow")
pip("albumentations")
pip("opencv-python-headless")
pip("PyYAML")

# ─── 2. IMPORTS ───────────────────────────────────────────────────────────────
import os, shutil, yaml, glob, cv2, random
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from roboflow import Roboflow

WORK_DIR = Path("/kaggle/working")
DATA_DIR = WORK_DIR / "finetune_data"
TRAIN_IMG = DATA_DIR / "images" / "train"
TRAIN_LBL = DATA_DIR / "labels" / "train"
VAL_IMG   = DATA_DIR / "images" / "val"
VAL_LBL   = DATA_DIR / "labels" / "val"

for d in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]:
    d.mkdir(parents=True, exist_ok=True)

print("✅ Directories created.")

# ─── 3. LOCATE EXISTING best.pt ───────────────────────────────────────────────
# Search standard Kaggle input locations
import glob as _glob
found = _glob.glob("/kaggle/input/**/*.pt", recursive=True)
if found:
    BASE_MODEL = found[0]
    print(f"✅ Found base model: {BASE_MODEL}")
else:
    # fallback: download yolov8m and you'll need to re-layer classes manually
    BASE_MODEL = "yolov8m.pt"
    print("⚠️  Could not find uploaded best.pt — using yolov8m.pt as fallback.")

# The existing model has 11 classes (indices 0-10).
# We will add 2 new classes: Dental Filling (11) and Dental Crown (12).
EXISTING_CLASSES = [
    "Occlusal Caries", "Proximal Caries", "Periapical Abscess",
    "Periapical Cyst", "Granuloma", "Apical Periodontitis",
    "Horizontal Bone Loss", "Vertical Bone Loss", "Root Canal Required",
    "Milk Tooth", "Healthy"
]
NEW_CLASSES = ["Dental Filling", "Dental Crown"]
ALL_CLASSES = EXISTING_CLASSES + NEW_CLASSES
NUM_CLASSES = len(ALL_CLASSES)   # 13
FILLING_ID  = 11
CROWN_ID    = 12

print(f"📊 Total classes after fine-tuning: {NUM_CLASSES}")

# ─── 4. DOWNLOAD NEW CLASS DATASETS FROM ROBOFLOW ─────────────────────────────
# REPLACE "YOUR_API_KEY" with your actual Roboflow API key before running.
ROBOFLOW_API_KEY = "YOUR_API_KEY"

def download_roboflow(api_key, workspace, project_slug, version, dest_dir, class_id):
    """Download dataset from Roboflow and copy to unified dataset with remapped class IDs."""
    rf = Roboflow(api_key=api_key)
    project  = rf.workspace(workspace).project(project_slug)
    dataset  = project.version(version).download("yolov8", location=str(dest_dir))
    print(f"✅ Downloaded: {project_slug}")
    return dest_dir

def copy_with_class_remap(src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir,
                           new_class_id, prefix, split_ratio=0.85):
    """
    Copy images + labels into the unified dataset folder,
    remapping all class IDs in label files to new_class_id.
    """
    imgs = list(Path(src_img_dir).glob("*.jpg")) + \
           list(Path(src_img_dir).glob("*.jpeg")) + \
           list(Path(src_img_dir).glob("*.png")) + \
           list(Path(src_img_dir).glob("*.bmp"))

    random.shuffle(imgs)
    n_train = int(len(imgs) * split_ratio)

    for idx, img_path in enumerate(imgs):
        lbl_path = Path(src_lbl_dir) / (img_path.stem + ".txt")

        is_train = idx < n_train
        dst_img  = (TRAIN_IMG if is_train else VAL_IMG)   / f"{prefix}_{img_path.name}"
        dst_lbl  = (TRAIN_LBL if is_train else VAL_LBL)   / f"{prefix}_{img_path.stem}.txt"

        # Apply CLAHE preprocessing before saving
        img = cv2.imread(str(img_path))
        if img is not None:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_eq  = clahe.apply(l)
            img   = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)
        cv2.imwrite(str(dst_img), img or cv2.imread(str(img_path)))

        # Remap class IDs in label file
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
            with open(dst_lbl, 'w') as f:
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        parts[0] = str(new_class_id)
                        f.write(" ".join(parts) + "\n")
        else:
            dst_lbl.touch()     # empty label (background sample)

    print(f"   → {n_train} train  /  {len(imgs)-n_train} val  [{prefix}]")


# ── Dental Filling ─────────────────────────────────────────────────────────────
print("\n📥 Downloading Dental Filling dataset...")
filling_dir = WORK_DIR / "filling_raw"
try:
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    proj = rf.workspace("dental-detection").project("dental-filling-detection")
    ds   = proj.version(1).download("yolov8", location=str(filling_dir))
    copy_with_class_remap(
        src_img_dir = filling_dir / "train" / "images",
        src_lbl_dir = filling_dir / "train" / "labels",
        dst_img_dir = TRAIN_IMG,
        dst_lbl_dir = TRAIN_LBL,
        new_class_id= FILLING_ID,
        prefix      = "fill"
    )
except Exception as e:
    print(f"⚠️  Filling download failed: {e}")
    print("   Will attempt alternate workspace slug if available.")

# ── Dental Crown ───────────────────────────────────────────────────────────────
print("\n📥 Downloading Dental Crown dataset...")
crown_dir = WORK_DIR / "crown_raw"
try:
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    proj = rf.workspace("dental-detection").project("dental-crown-detection")
    ds   = proj.version(1).download("yolov8", location=str(crown_dir))
    copy_with_class_remap(
        src_img_dir = crown_dir / "train" / "images",
        src_lbl_dir = crown_dir / "train" / "labels",
        dst_img_dir = TRAIN_IMG,
        dst_lbl_dir = TRAIN_LBL,
        new_class_id= CROWN_ID,
        prefix      = "crown"
    )
except Exception as e:
    print(f"⚠️  Crown download failed: {e}")

# ─── 5. PRINT DATA SUMMARY ────────────────────────────────────────────────────
n_train = len(list(TRAIN_IMG.glob("*.*")))
n_val   = len(list(VAL_IMG.glob("*.*")))
print(f"\n📊 Dataset Summary:")
print(f"   Train images : {n_train}")
print(f"   Val images   : {n_val}")

# ─── 6. WRITE data.yaml ───────────────────────────────────────────────────────
data_yaml = {
    'path'  : str(DATA_DIR),
    'train' : 'images/train',
    'val'   : 'images/val',
    'nc'    : NUM_CLASSES,
    'names' : ALL_CLASSES,
}
yaml_path = DATA_DIR / "data.yaml"
with open(yaml_path, 'w', encoding='utf-8') as f:
    yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
print(f"\n✅ data.yaml written:\n{yaml.dump(data_yaml, default_flow_style=False)}")

# ─── 7. FINE-TUNE WITH FROZEN BACKBONE ────────────────────────────────────────
print("\n🚀 Starting fine-tuning...")

model = YOLO(BASE_MODEL)

results = model.train(
    data        = str(yaml_path),
    epochs      = 100,
    imgsz       = 640,
    batch       = 16,
    device      = 0,            # P100 GPU
    optimizer   = "AdamW",
    lr0         = 0.001,
    lrf         = 0.01,
    momentum    = 0.937,
    weight_decay= 0.0005,
    warmup_epochs = 3,
    cos_lr      = True,
    augment     = True,         # standard augmentation
    hsv_h       = 0.015,        # hue augmentation
    hsv_s       = 0.7,
    hsv_v       = 0.4,
    flipud      = 0.1,
    fliplr      = 0.5,
    mosaic      = 1.0,
    mixup       = 0.15,
    freeze      = 20,           # Freeze first 20 backbone layers
    patience    = 30,
    save_period = 10,
    project     = "/kaggle/working/runs",
    name        = "finetune_filling_crown",
    exist_ok    = True,
    verbose     = True,
)

# ─── 8. VALIDATE AND REPORT ───────────────────────────────────────────────────
print("\n🔍 Running validation on fine-tuned model...")
best_path = "/kaggle/working/runs/finetune_filling_crown/weights/best.pt"
best_model = YOLO(best_path)

val_results = best_model.val(
    data   = str(yaml_path),
    imgsz  = 640,
    conf   = 0.001,
    iou    = 0.6,
    device = 0,
)

print(f"\n✅ Validation mAP@50   : {val_results.box.map50:.4f}")
print(f"   Validation mAP@50-95: {val_results.box.map:.4f}")
print(f"   Precision            : {val_results.box.mp:.4f}")
print(f"   Recall               : {val_results.box.mr:.4f}")

# ─── 9. COPY BEST MODEL TO TOP-LEVEL ─────────────────────────────────────────
output_model = "/kaggle/working/best_finetuned.pt"
shutil.copy(best_path, output_model)
print(f"\n🎉 Done! Fine-tuned model saved to: {output_model}")
print("   Download this file and rename it to: dental_yolov8.pt")
print("   Place it inside your Django project's: models/ folder")
