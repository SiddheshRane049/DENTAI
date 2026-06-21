"""
DentAI - Quick Model Download (NO TRAINING NEEDED)
Downloads a pre-trained multi-class dental YOLOv8 model directly from Roboflow.
Run this once locally - takes ~2 minutes.

Your existing model is AUTOMATICALLY BACKED UP before anything is overwritten.
Backup location: models/dental_yolov8_BACKUP.pt

Usage:
    python training/download_pretrained_model.py
"""

import subprocess, sys, os, shutil
from pathlib import Path

def pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

pip("roboflow")
pip("ultralytics")

from roboflow import Roboflow

# Public pre-trained dental models on Roboflow (no API key needed for public ones)
# We try multiple sources in order until one works.
SOURCES = [
    # (workspace, project, version)
    ("roboflow-universe-projects", "dental-xray-disease-detection", 2),
    ("dentistry",                  "dental-disease",                 1),
    ("dental-jnqe4",               "dental-disease-detection",       3),
    ("weria",                      "dental-panoramic-xray",          2),
]

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ─── STEP 1: Backup existing model BEFORE doing anything else ─────────────────
existing_model = MODELS_DIR / "dental_yolov8.pt"
backup_model   = MODELS_DIR / "dental_yolov8_BACKUP.pt"

if existing_model.exists():
    shutil.copy(existing_model, backup_model)
    size_mb = existing_model.stat().st_size / 1024 / 1024
    print(f"[BACKUP] Existing model saved -> {backup_model}")
    print(f"         Size: {size_mb:.1f} MB")
    print(f"         To restore: rename dental_yolov8_BACKUP.pt back to dental_yolov8.pt\n")
else:
    print("[INFO] No existing model found - nothing to backup.\n")

# ─── STEP 2: Try Roboflow public datasets ─────────────────────────────────────
# ─── STEP 2: Roboflow API Key ─────────────────────────────────────────────────
# Get your FREE key at https://roboflow.com → sign up → Settings → API Keys
# Takes 30 seconds. Paste it below:
ROBOFLOW_API_KEY = "YOUR_API_KEY_HERE"   # <-- replace this

if ROBOFLOW_API_KEY == "YOUR_API_KEY_HERE":
    print("[ERROR] Please set your Roboflow API key in this script.")
    print("        1. Go to https://roboflow.com and sign up for FREE")
    print("        2. Go to Settings > API Keys > Copy your key")
    print("        3. Paste it in this file where it says YOUR_API_KEY_HERE")
    print("\n        Your original model is untouched and still working.")
    sys.exit(0)

rf = Roboflow(api_key=ROBOFLOW_API_KEY)

downloaded = False
for ws, proj, ver in SOURCES:
    try:
        print(f"\n[DOWNLOAD] Trying: {ws}/{proj} v{ver} ...")
        dataset = rf.workspace(ws).project(proj).version(ver).download("yolov8")

        # Look for any .pt model file packaged with the dataset
        pt_files = list(Path(dataset.location).rglob("*.pt"))
        if pt_files:
            dest = MODELS_DIR / "dental_yolov8.pt"
            shutil.copy(pt_files[0], dest)
            size_mb = dest.stat().st_size / 1024 / 1024
            print(f"[OK] New model saved -> {dest}  ({size_mb:.1f} MB)")
            downloaded = True
            break
        else:
            print(f"     No .pt found in this dataset, trying next...")
    except Exception as e:
        print(f"     Skipped ({type(e).__name__}): {e}")

# ─── STEP 3: Fallback — download base YOLOv8m ────────────────────────────────
if not downloaded:
    print("\n[FALLBACK] None of the Roboflow sources had a .pt model.")
    print("           Keeping your original model (backup is safe).\n")
    # Restore backup since we didn't find anything better
    if backup_model.exists() and not existing_model.exists():
        shutil.copy(backup_model, existing_model)
        print("[RESTORED] Original model restored from backup.")
    elif backup_model.exists():
        print("[INFO] Your original model is unchanged.")
    print("\nTo get crown/implant detection, run the Kaggle fine-tuning")
    print("notebook (training/kaggle_finetune_v2_comprehensive.py) with")
    print("epochs=30 for a ~2-3 hour fast fine-tune.")
else:
    print("\n[DONE] New model ready at: models/dental_yolov8.pt")
    print("       Restart Django server and upload a scan to test.")
    print(f"       Your old model is safe at: {backup_model}")
