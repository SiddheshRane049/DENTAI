"""
DentAI Training Pipeline — Step 2: Format Converter
=====================================================
Converts all downloaded datasets to YOLO format (txt bounding boxes).

Input formats handled:
  - yolov8    : Already in YOLO format (images/ + labels/ structure)
  - coco_json : COCO annotations.json → per-image .txt files
  - voc_xml   : Pascal VOC XML → per-image .txt files
  - csv_bbox  : CSV with columns [filename, xmin, ymin, xmax, ymax, class] → .txt
  - jpg_only  : Images without annotations (skipped — need manual labeling)

Class remapping:
  Each dataset has a `diseases` list that maps the SOURCE class IDs
  to our 11-class DentAI target schema.

Output:
  training/converted/<dataset_name>/
    images/   (all JPEG/PNG files copied here)
    labels/   (YOLO .txt files — one per image)
"""

import os
import sys
import io
import json
import glob
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_DIR       = Path(__file__).resolve().parent
RAW_DATA_DIR   = BASE_DIR / "raw_datasets"
CONVERTED_DIR  = BASE_DIR / "converted"
MANIFEST_PATH  = BASE_DIR / "dataset_manifest.json"

CONVERTED_DIR.mkdir(parents=True, exist_ok=True)


# ─── DentAI Class Schema ─────────────────────────────────────────────────────
# Canonical class IDs used throughout training
DENTAI_CLASSES = {
    0:  "occlusal_caries",
    1:  "proximal_caries",
    2:  "periapical_abscess",
    3:  "periapical_cyst",
    4:  "granuloma",
    5:  "apical_periodontitis",
    6:  "horizontal_bone_loss",
    7:  "vertical_bone_loss",
    8:  "root_canal_required",
    9:  "milk_tooth",
    10: "healthy",
}

# ─── Dataset-specific class name → DentAI class ID remaps ───────────────────
# Each dataset uses its own class names; we map them here.
SOURCE_CLASS_ALIASES = {
    # Caries
    "caries":                 0,
    "occlusal caries":        0,
    "occlusal_caries":        0,
    "dental caries":          0,
    "decay":                  0,
    "proximal caries":        1,
    "proximal_caries":        1,
    "interproximal caries":   1,

    # Periapical lesions
    "periapical abscess":     2,
    "periapical_abscess":     2,
    "abscess":                2,
    "periapical lesion":      2,
    "periapical_lesion":      2,
    "apical lesion":          2,

    # Cyst
    "cyst":                   3,
    "periapical cyst":        3,
    "periapical_cyst":        3,
    "radicular cyst":         3,
    "jaw cyst":               3,

    # Granuloma
    "granuloma":              4,
    "periapical granuloma":   4,
    "periapical_granuloma":   4,

    # Periodontitis
    "periodontitis":          5,
    "apical periodontitis":   5,
    "apical_periodontitis":   5,
    "periapical periodontitis": 5,

    # Bone loss
    "bone loss":              6,
    "bone_loss":              6,
    "horizontal bone loss":   6,
    "horizontal_bone_loss":   6,
    "alveolar bone loss":     6,
    "vertical bone loss":     7,
    "vertical_bone_loss":     7,
    "angular bone loss":      7,

    # Root canal
    "root canal":             8,
    "root_canal":             8,
    "root canal treatment":   8,
    "endodontic":             8,
    "endo":                   8,

    # Milk tooth
    "milk tooth":             9,
    "milk_tooth":             9,
    "deciduous":              9,
    "primary tooth":          9,
    "baby tooth":             9,

    # Healthy / background
    "healthy":                10,
    "normal":                 10,
    "tooth":                  10,   # Generic tooth = treat as healthy region
}


def resolve_class(raw_class_name: str) -> int | None:
    """
    Map a source class name to a DentAI class ID.
    Returns None if the class should be skipped.
    """
    key = str(raw_class_name).lower().strip()
    return SOURCE_CLASS_ALIASES.get(key, None)


# ─── Converters ───────────────────────────────────────────────────────────────

def convert_yolov8(src_dir: Path, out_dir: Path, remap: list[int]):
    out_dir.mkdir(parents=True, exist_ok=True)
    img_out  = out_dir / "images"
    lbl_out  = out_dir / "labels"
    img_out.mkdir(exist_ok=True)
    lbl_out.mkdir(exist_ok=True)

    # Find all 'images' directories recursively
    images_dirs = list(src_dir.rglob("images"))
    if not images_dirs:
        # Fallback if images are just in the root alongside labels
        images_dirs = [src_dir]
        
    converted = 0
    for idir in images_dirs:
        # Try to find corresponding labels dir
        ldir = idir.parent / "labels" if idir.name == "images" else src_dir / "labels"
        if not ldir.exists():
            continue

        for img_path in list(idir.glob("*.jpg")) + list(idir.glob("*.png")):
            lbl_path = ldir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            # Remap class IDs
            new_lines = []
            for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    src_cls = int(float(parts[0]))
                except ValueError:
                    continue
                
                # Use remap list if in range, else try alias map
                if src_cls < len(remap):
                    tgt_cls = remap[src_cls]
                else:
                    tgt_cls = src_cls
                new_lines.append(f"{tgt_cls} " + " ".join(parts[1:]))

            new_lbl = lbl_out / (img_path.stem + ".txt")
            new_lbl.write_text("\n".join(new_lines), encoding="utf-8")
            shutil.copy(img_path, img_out / img_path.name)
            converted += 1

    print(f"    ✓ YOLOv8 converted: {converted} images")
    return converted


def convert_coco_json(src_dir: Path, out_dir: Path, diseases: list[int]):
    out_dir.mkdir(parents=True, exist_ok=True)
    img_out  = out_dir / "images"
    lbl_out  = out_dir / "labels"
    img_out.mkdir(exist_ok=True)
    lbl_out.mkdir(exist_ok=True)

    json_files = list(src_dir.rglob("*.json"))
    converted  = 0

    for json_file in json_files:
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        if "annotations" not in data or "images" not in data:
            continue

        img_map = {img["id"]: img for img in data["images"]}

        cat_map = {}
        for cat in data.get("categories", []):
            tgt = resolve_class(cat.get("name", ""))
            if tgt is not None:
                cat_map[cat["id"]] = tgt

        img_anns = {}
        for ann in data["annotations"]:
            iid = ann.get("image_id")
            if iid is not None:
                img_anns.setdefault(iid, []).append(ann)

        for img_id, img_info in img_map.items():
            anns = img_anns.get(img_id, [])
            if not anns:
                continue

            img_w = img_info.get("width", 1)
            img_h = img_info.get("height", 1)

            fname     = img_info.get("file_name", "")
            img_path  = _find_image(src_dir, fname)
            if not img_path:
                continue

            lines = []
            for ann in anns:
                cat_id = ann.get("category_id")
                cls = cat_map.get(cat_id)
                if cls is None:
                    continue

                if "bbox" not in ann:
                    continue
                x, y, bw, bh = ann["bbox"]
                cx = (x + bw / 2) / img_w
                cy = (y + bh / 2) / img_h
                nw = bw / img_w
                nh = bh / img_h

                cx, cy, nw, nh = (max(0, min(1, v)) for v in (cx, cy, nw, nh))
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            if lines:
                stem    = Path(fname).stem
                lbl_out.joinpath(stem + ".txt").write_text("\n".join(lines), encoding="utf-8")
                shutil.copy(img_path, img_out / Path(fname).name)
                converted += 1

    print(f"    ✓ COCO JSON converted: {converted} images")
    return converted


def convert_voc_xml(src_dir: Path, out_dir: Path, diseases: list[int]):
    """Pascal VOC XML format → YOLO txt."""
    out_dir.mkdir(parents=True, exist_ok=True)
    img_out = out_dir / "images"
    lbl_out = out_dir / "labels"
    img_out.mkdir(exist_ok=True)
    lbl_out.mkdir(exist_ok=True)

    xml_files  = list(src_dir.rglob("*.xml"))
    converted  = 0

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
        except ET.ParseError:
            continue

        # Image dimensions
        size = root.find("size")
        if size is None:
            continue
        img_w = int(size.findtext("width", 0))
        img_h = int(size.findtext("height", 0))
        if img_w == 0 or img_h == 0:
            continue

        fname    = root.findtext("filename", "")
        img_path = _find_image(src_dir, fname)
        if not img_path:
            continue

        lines = []
        for obj in root.findall("object"):
            class_name = obj.findtext("name", "").strip()
            cls = resolve_class(class_name)
            if cls is None:
                continue

            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            x1 = float(bndbox.findtext("xmin", 0))
            y1 = float(bndbox.findtext("ymin", 0))
            x2 = float(bndbox.findtext("xmax", 0))
            y2 = float(bndbox.findtext("ymax", 0))

            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            nw = (x2 - x1) / img_w
            nh = (y2 - y1) / img_h

            cx, cy, nw, nh = (max(0, min(1, v)) for v in (cx, cy, nw, nh))
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if lines:
            stem = Path(fname).stem
            lbl_out.joinpath(stem + ".txt").write_text("\n".join(lines))
            shutil.copy(img_path, img_out / Path(fname).name)
            converted += 1

    print(f"    ✓ VOC XML converted: {converted} images")
    return converted


def convert_csv_bbox(src_dir: Path, out_dir: Path, diseases: list[int]):
    """
    CSV format with columns:
    filename, xmin, ymin, xmax, ymax, class_name [, width, height]
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img_out = out_dir / "images"
    lbl_out = out_dir / "labels"
    img_out.mkdir(exist_ok=True)
    lbl_out.mkdir(exist_ok=True)

    csv_files  = list(src_dir.rglob("*.csv"))
    converted  = 0

    for csv_file in csv_files:
        import csv
        rows_by_file = {}
        try:
            with open(csv_file, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fname = row.get("filename") or row.get("file") or row.get("image")
                    if not fname:
                        continue
                    rows_by_file.setdefault(fname, []).append(row)
        except Exception as e:
            print(f"      ⚠ CSV parse error: {csv_file.name}: {e}")
            continue

        for fname, rows in rows_by_file.items():
            img_path = _find_image(src_dir, fname)
            if not img_path:
                continue

            # Get image dimensions
            try:
                img = Image.open(img_path)
                img_w, img_h = img.size
                img.close()
            except Exception:
                continue

            lines = []
            for row in rows:
                class_name = (
                    row.get("class") or row.get("label") or
                    row.get("class_name") or row.get("disease") or ""
                ).strip()
                cls = resolve_class(class_name)
                if cls is None:
                    continue

                try:
                    x1 = float(row.get("xmin") or row.get("x1") or 0)
                    y1 = float(row.get("ymin") or row.get("y1") or 0)
                    x2 = float(row.get("xmax") or row.get("x2") or 0)
                    y2 = float(row.get("ymax") or row.get("y2") or 0)
                except (ValueError, TypeError):
                    continue

                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                nw = (x2 - x1) / img_w
                nh = (y2 - y1) / img_h
                cx, cy, nw, nh = (max(0, min(1, v)) for v in (cx, cy, nw, nh))
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            if lines:
                stem = Path(fname).stem
                lbl_out.joinpath(stem + ".txt").write_text("\n".join(lines))
                shutil.copy(img_path, img_out / Path(fname).name)
                converted += 1

    print(f"    ✓ CSV converted: {converted} images")
    return converted


def _find_image(base_dir: Path, filename: str) -> Path | None:
    """Search recursively for an image file by name."""
    target = Path(filename).name
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        stem = Path(filename).stem
        for found in base_dir.rglob(f"{stem}{ext}"):
            return found
    return None


# ─── Dispatch ─────────────────────────────────────────────────────────────────

FORMAT_DISPATCH = {
    "yolov8":    convert_yolov8,
    "coco_json": convert_coco_json,
    "voc_xml":   convert_voc_xml,
    "csv_bbox":  convert_csv_bbox,
    "jpg_only":  None,     # Skip — images-only datasets
}


def main():
    print("=" * 60)
    print("  DentAI — Step 2: Converting Datasets to YOLO Format")
    print("=" * 60)

    if not MANIFEST_PATH.exists():
        print("✗ dataset_manifest.json not found. Run Step 1 first.")
        return

    manifest = json.loads(MANIFEST_PATH.read_text())
    totals   = {}

    # Get unique dataset names from manifest
    ds_names = [k for k in manifest if not k.endswith("__fmt") and not k.endswith("__diseases")]

    for name in ds_names:
        src_path = Path(manifest[name])
        fmt      = manifest.get(f"{name}__fmt", "yolov8")
        diseases = manifest.get(f"{name}__diseases", [0])

        print(f"\n[{fmt.upper()}] {name}")

        out_dir  = CONVERTED_DIR / name
        convert_fn = FORMAT_DISPATCH.get(fmt)

        if convert_fn is None:
            print(f"  ⚠ Skipping {fmt} (images only — annotate manually)")
            continue

        if not src_path.exists():
            print(f"  ✗ Source not found: {src_path}")
            continue

        # Build remap list for yolov8 format
        remap = list(range(max(diseases) + 2))  # identity by default
        if fmt == "yolov8":
            # For Roboflow datasets: source class 0 → diseases[0], etc.
            for i, tgt in enumerate(diseases):
                if i < len(remap):
                    remap[i] = tgt

        try:
            count = convert_fn(src_path, out_dir, remap if fmt == "yolov8" else diseases)
            totals[name] = count
        except Exception as e:
            print(f"  ✗ Conversion error: {e}")

    total_imgs = sum(totals.values())
    print("\n" + "=" * 60)
    print(f"  ✅ Conversion complete: {total_imgs} total images")
    for name, cnt in totals.items():
        print(f"    {name:<40} {cnt:>5} images")
    print("  → Run next: python training/3_merge_and_split.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
