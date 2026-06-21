"""
DentAI Training Pipeline — Step 3: Merge & Split Dataset
==========================================================
Merges all converted datasets into a single unified dataset
with train / val / test splits and generates data.yaml.

Features:
  - De-duplication by image hash (prevents duplicates across datasets)
  - Stratified split (80% train, 15% val, 5% test)
  - Class distribution analysis / per-class stats
  - Augmentation-aware file naming (preserves dataset source)
  - Generates data.yaml for YOLOv8 training
"""

import os
import sys
import io
import json
import shutil
import hashlib
import random
from pathlib import Path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_DIR      = Path(__file__).resolve().parent
CONVERTED_DIR = BASE_DIR / "converted"
MERGED_DIR    = BASE_DIR / "merged_dataset"

TRAIN_RATIO   = 0.80
VAL_RATIO     = 0.15
TEST_RATIO    = 0.05

SEED = 42
random.seed(SEED)

# DentAI class names (must match settings.py DENTAL_DISEASE_CLASSES order)
CLASS_NAMES = [
    "occlusal_caries",         # 0
    "proximal_caries",         # 1
    "periapical_abscess",      # 2
    "periapical_cyst",         # 3
    "granuloma",               # 4
    "apical_periodontitis",    # 5
    "horizontal_bone_loss",    # 6
    "vertical_bone_loss",      # 7
    "root_canal_required",     # 8
    "milk_tooth",              # 9
    "healthy",                 # 10
]


def image_hash(img_path: Path) -> str:
    """MD5 hash of image content for de-duplication."""
    h = hashlib.md5()
    with open(img_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_classes_in_label(lbl_path: Path) -> list[int]:
    """Return list of class IDs present in a YOLO label file."""
    classes = set()
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.split()
        if parts:
            classes.add(int(parts[0]))
    return list(classes)


def main():
    print("=" * 60)
    print("  DentAI — Step 3: Merging & Splitting Dataset")
    print("=" * 60)

    # ── Collect all converted image-label pairs ────────────────────────────
    all_pairs  = []   # List of (img_path, lbl_path, dataset_name)
    seen_hashes= set()

    for ds_dir in sorted(CONVERTED_DIR.iterdir()):
        if not ds_dir.is_dir():
            continue
        img_dir = ds_dir / "images"
        lbl_dir = ds_dir / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            continue

        added = 0
        skipped_dup = 0
        skipped_no_lbl = 0

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists() or lbl_path.stat().st_size == 0:
                skipped_no_lbl += 1
                continue

            # De-duplicate by image hash
            h = image_hash(img_path)
            if h in seen_hashes:
                skipped_dup += 1
                continue
            seen_hashes.add(h)

            all_pairs.append((img_path, lbl_path, ds_dir.name))
            added += 1

        print(f"  {ds_dir.name:<40} +{added:>4} images  "
              f"(dup:{skipped_dup}, no_label:{skipped_no_lbl})")

    total = len(all_pairs)
    print(f"\n  Total unique annotated images: {total}")

    if total == 0:
        print("  ✗ No images found. Run Step 2 first.")
        return

    # ── Stratified split by dominant class ────────────────────────────────
    # Group by the first class in each label file
    by_class = defaultdict(list)
    for pair in all_pairs:
        classes = get_classes_in_label(pair[1])
        dominant = classes[0] if classes else 10
        by_class[dominant].append(pair)

    train_set, val_set, test_set = [], [], []

    for cls_id, pairs in by_class.items():
        random.shuffle(pairs)
        n = len(pairs)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        train_set.extend(pairs[:n_train])
        val_set.extend(pairs[n_train:n_train + n_val])
        test_set.extend(pairs[n_train + n_val:])

    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    print(f"\n  Split: train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")

    # ── Copy to merged_dataset/ ────────────────────────────────────────────
    for split_name, split_pairs in [("train", train_set), ("val", val_set), ("test", test_set)]:
        img_out = MERGED_DIR / split_name / "images"
        lbl_out = MERGED_DIR / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for idx, (img_path, lbl_path, ds_name) in enumerate(split_pairs):
            # Unique filename: <dataset>_<index><ext>
            ext      = img_path.suffix
            new_stem = f"{ds_name[:20]}_{idx:05d}"
            shutil.copy(img_path, img_out / (new_stem + ext))
            shutil.copy(lbl_path, lbl_out / (new_stem + ".txt"))

        print(f"  ✓ {split_name}: {len(split_pairs)} files written")

    # ── Class distribution report ──────────────────────────────────────────
    print("\n  Class Distribution (training set):")
    class_counts = defaultdict(int)
    for _, lbl_path, _ in train_set:
        for cls in get_classes_in_label(lbl_path):
            class_counts[cls] += 1

    for cls_id, name in enumerate(CLASS_NAMES):
        count = class_counts.get(cls_id, 0)
        bar   = "█" * min(40, count // 10)
        print(f"    [{cls_id:2d}] {name:<28} {count:>5}  {bar}")

    # ── Generate data.yaml ────────────────────────────────────────────────
    data_yaml = MERGED_DIR / "data.yaml"
    yaml_content = f"""# DentAI — YOLOv8 Training Dataset Configuration
# Generated by training/3_merge_and_split.py
# Total images: {total} | Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}

path: {str(MERGED_DIR).replace(chr(92), '/')}   # dataset root dir
train: train/images
val:   val/images
test:  test/images

nc: {len(CLASS_NAMES)}   # number of classes

names:
{chr(10).join(f'  {i}: {name}' for i, name in enumerate(CLASS_NAMES))}

# DentAI disease class info
# 0  occlusal_caries      - Occlusal surface cavities
# 1  proximal_caries      - Between-tooth cavities
# 2  periapical_abscess   - Root-tip bacterial infection
# 3  periapical_cyst      - Epithelial-lined lesion at root
# 4  granuloma            - Chronic inflammatory lesion
# 5  apical_periodontitis - Inflammation at root tip
# 6  horizontal_bone_loss - Even alveolar bone resorption
# 7  vertical_bone_loss   - Angular alveolar bone defect
# 8  root_canal_required  - Pulp involvement needs RCT
# 9  milk_tooth           - Primary/deciduous tooth
# 10 healthy              - No pathology detected
"""
    data_yaml.write_text(yaml_content)
    print(f"\n  ✅ data.yaml written to: {data_yaml}")
    print("  → Run next: python training/4_train_yolov8.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
