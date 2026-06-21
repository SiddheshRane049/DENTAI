"""
╔══════════════════════════════════════════════════════════════════════╗
║  DentAI — Comprehensive Fine-Tuning: 15-Class Dental Detection      ║
║  Adds: Implant, Impacted Tooth, Calculus, Root Fracture + more      ║
║  Run on Kaggle Notebook with P100 GPU                               ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import subprocess, sys

def pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkg.split())

pip("ultralytics")
pip("albumentations")
pip("opencv-python-headless")
pip("PyYAML")

import os, shutil, yaml, glob, cv2, random
import numpy as np
from pathlib import Path
from ultralytics import YOLO

WORK_DIR = Path("/kaggle/working")
DATA_DIR = WORK_DIR / "dentai_v2_data"
TRAIN_IMG = DATA_DIR / "images" / "train"
TRAIN_LBL = DATA_DIR / "labels" / "train"
VAL_IMG   = DATA_DIR / "images" / "val"
VAL_LBL   = DATA_DIR / "labels" / "val"

for d in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]:
    d.mkdir(parents=True, exist_ok=True)

print("✅ Directories created.")

BASE_MODEL = "yolov8x.pt"

ALL_CLASSES = [
    "Occlusal Caries", "Proximal Caries", "Periapical Abscess",
    "Periapical Cyst", "Granuloma", "Apical Periodontitis",
    "Horizontal Bone Loss", "Vertical Bone Loss", "Root Canal Required",
    "Milk Tooth", "Healthy",
    "Dental Filling", "Dental Crown", "Dental Implant", "Impacted Tooth",
    "Calculus", "Root Fracture", "Retained Root"
]
NUM_CLASSES = len(ALL_CLASSES)

print("\n🔍 Scanning /kaggle/input for datasets you added...")

input_dir = Path("/kaggle/input")
all_images = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]

all_txt = list(input_dir.rglob("*.txt"))
label_dict = {p.stem: p for p in all_txt if not p.name.lower().startswith(("readme", "license", "classes"))}

valid_pairs = []
for img_p in all_images:
    if img_p.stem in label_dict:
        valid_pairs.append((img_p, label_dict[img_p.stem]))

if not valid_pairs:
    raise ValueError("🚨 FATAL ERROR: No datasets found! Please click '+ Add Data' in Kaggle, search for 'Dental', and add a dataset.")

copied = 0
for img_p, lbl_p in valid_pairs:
    is_train = random.random() < 0.85
    dst_img = (TRAIN_IMG if is_train else VAL_IMG) / f"{copied}_{img_p.name}"
    dst_lbl = (TRAIN_LBL if is_train else VAL_LBL) / f"{copied}_{img_p.stem}.txt"
    shutil.copy(str(img_p), str(dst_img))
    shutil.copy(str(lbl_p), str(dst_lbl))
    copied += 1

print(f"✅ Automatically merged {copied} images into unified dataset!")

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

print("\n🚀 Starting fine-tuning...")

model = YOLO(BASE_MODEL)
results = model.train(
    data        = str(yaml_path),
    epochs      = 150,
    imgsz       = 1024,
    batch       = 16,
    device      = 0,
    optimizer   = "AdamW",
    lr0         = 0.001,
    patience    = 40,
    project     = "/kaggle/working/runs",
    name        = "dentai_v2_comprehensive",
    exist_ok    = True,
    verbose     = True,
)

best_path = "/kaggle/working/runs/dentai_v2_comprehensive/weights/best.pt"
output_model = "/kaggle/working/structures_yolov8.pt"
shutil.copy(best_path, output_model)
print(f"\n🎉 Done! Model saved to: {output_model}")
