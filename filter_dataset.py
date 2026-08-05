"""
YOLOv8 Dataset Filter Script
Filters a YOLOv8 dataset by keeping only specified classes, remapping class IDs,
removing images with zero remaining annotations, preserving split structures,
and generating a new data.yaml file with full output validation.
"""

import os
import shutil
import sys
from collections import Counter
from pathlib import Path

# Try importing PyYAML, fall back to manual YAML writing if not available
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ==========================================
# CONFIGURATION
# ==========================================
INPUT_DATASET_DIR = Path(r"D:\DentAI-missing-classes-v1")
OUTPUT_DATASET_DIR = Path(r"D:\DentAI-filtered")

SPLITS = ["train", "valid", "test"]

# Original Class ID -> (New Class ID, Class Name)
CLASS_MAPPING = {
    0: 0,   # Bone Loss
    2: 1,   # Crown
    4: 2,   # Filling
    9: 3,   # Missing Teeth
    10: 4,  # Periapical Lesion
    12: 5,  # Primary Teeth
    13: 6,  # Retained Root
    15: 7,  # Root Piece
    23: 8,  # Impacted Tooth
}

NEW_CLASS_NAMES = {
    0: "Bone Loss",
    1: "Crown",
    2: "Filling",
    3: "Missing Teeth",
    4: "Periapical Lesion",
    5: "Primary Teeth",
    6: "Retained Root",
    7: "Root Piece",
    8: "Impacted Tooth",
}

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def create_directory_structure(output_dir: Path, splits: list[str]) -> None:
    """Create the destination folder structure for images and labels."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        for split in splits:
            (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[ERROR] Failed to create directory structure in {output_dir}: {e}")
        raise


def parse_and_filter_label_file(label_path: Path, class_mapping: dict[int, int]) -> tuple[list[str], Counter, int]:
    """
    Read a YOLO label file line-by-line.
    Filter annotations belonging to selected original class IDs and remap them.

    Returns:
        kept_lines: list of formatted string lines for the new label file
        kept_counts: Counter mapping new_class_id -> count
        removed_count: number of annotations removed
    """
    kept_lines = []
    kept_counts = Counter()
    removed_count = 0

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            parts = line_str.split()
            if not parts:
                continue

            try:
                orig_class_id = int(parts[0])
            except ValueError:
                # Handle malformed line gracefully
                continue

            if orig_class_id in class_mapping:
                new_class_id = class_mapping[orig_class_id]
                parts[0] = str(new_class_id)
                new_line = " ".join(parts) + "\n"
                kept_lines.append(new_line)
                kept_counts[new_class_id] += 1
            else:
                removed_count += 1

    except Exception as e:
        print(f"[WARNING] Error reading label file {label_path}: {e}")

    return kept_lines, kept_counts, removed_count


def generate_data_yaml(output_dir: Path, class_names: dict[int, str]) -> None:
    """Generate the data.yaml file in the output dataset root directory."""
    yaml_path = output_dir / "data.yaml"
    nc = len(class_names)

    if HAS_YAML:
        yaml_content = {
            "train": "../train/images",
            "val": "../valid/images",
            "test": "../test/images",
            "nc": nc,
            "names": {k: v for k, v in sorted(class_names.items())}
        }
        try:
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=False)
            return
        except Exception as e:
            print(f"[WARNING] PyYAML dump failed ({e}), falling back to manual writing.")

    # Manual YAML formatting fallback matching requirement exactly
    yaml_lines = [
        "train: ../train/images",
        "val: ../valid/images",
        "test: ../test/images",
        "",
        f"nc: {nc}",
        "",
        "names:"
    ]
    for cid in sorted(class_names.keys()):
        yaml_lines.append(f"  {cid}: {class_names[cid]}")
    yaml_lines.append("")

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines))


def validate_output_dataset(output_dir: Path, splits: list[str], expected_nc: int) -> bool:
    """
    Validate output dataset according to requirement 10:
    - Every copied image must have a label file.
    - Every copied label file must contain at least one annotation.
    - Every class ID in the output labels must be between 0 and expected_nc - 1.
    - Print 'Dataset validation passed' if everything is correct.
    """
    print("\n" + "=" * 50)
    print("RUNNING DATASET VALIDATION")
    print("=" * 50)

    validation_errors = []

    for split in splits:
        img_dir = output_dir / split / "images"
        lbl_dir = output_dir / split / "labels"

        images = {f.stem: f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS}
        labels = {f.stem: f for f in lbl_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"}

        # 1. Every copied image must have a label file
        for stem, img_path in images.items():
            if stem not in labels:
                validation_errors.append(f"[{split}] Image missing label file: {img_path.name}")

        # 2. Every label file must correspond to an image, contain >=1 annotation, and class IDs in [0, nc-1]
        for stem, lbl_path in labels.items():
            if stem not in images:
                validation_errors.append(f"[{split}] Label file missing image: {lbl_path.name}")

            try:
                with open(lbl_path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]

                if len(lines) == 0:
                    validation_errors.append(f"[{split}] Empty label file found: {lbl_path.name}")

                for line in lines:
                    parts = line.split()
                    try:
                        cid = int(parts[0])
                        if cid < 0 or cid >= expected_nc:
                            validation_errors.append(f"[{split}] Invalid class ID {cid} in {lbl_path.name}")
                    except (ValueError, IndexError):
                        validation_errors.append(f"[{split}] Malformed line in {lbl_path.name}: '{line}'")
            except Exception as e:
                validation_errors.append(f"[{split}] Error reading {lbl_path.name} during validation: {e}")

    # Check data.yaml exists
    yaml_path = output_dir / "data.yaml"
    if not yaml_path.exists():
        validation_errors.append("data.yaml does not exist in output root directory.")

    if validation_errors:
        print("[FAIL] Validation failed with the following errors:")
        for err in validation_errors[:20]:
            print(f"  - {err}")
        if len(validation_errors) > 20:
            print(f"  ... and {len(validation_errors) - 20} more errors.")
        return False
    else:
        print("Dataset validation passed")
        return True


def filter_dataset(
    input_dir: Path,
    output_dir: Path,
    splits: list[str],
    class_mapping: dict[int, int],
    class_names: dict[int, str]
) -> None:
    """Main dataset filtering function."""
    print("=" * 60)
    print("STARTING YOLOV8 DATASET FILTERING")
    print(f"Input Dataset:  {input_dir}")
    print(f"Output Dataset: {output_dir}")
    print("=" * 60)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    # Ensure output dataset directory is initialized
    create_directory_structure(output_dir, splits)

    # Statistics tracking
    images_copied_per_split = {split: 0 for split in splits}
    total_kept_per_class = Counter()
    total_removed_annotations = 0
    total_skipped_images = 0

    for split in splits:
        split_in_img = input_dir / split / "images"
        split_in_lbl = input_dir / split / "labels"

        split_out_img = output_dir / split / "images"
        split_out_lbl = output_dir / split / "labels"

        if not split_in_img.exists() or not split_in_lbl.exists():
            print(f"[WARNING] Split folder missing images or labels: {split}")
            continue

        print(f"\nProcessing split: '{split}'...")

        # Build fast lookup for input images by stem
        image_files = {
            f.stem: f for f in split_in_img.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        }

        # Process all label files in the split
        label_files = list(split_in_lbl.glob("*.txt"))

        for lbl_path in label_files:
            stem = lbl_path.stem

            if stem not in image_files:
                # Label file without matching image - skip
                continue

            img_path = image_files[stem]

            # Filter label annotations
            kept_lines, kept_counts, removed_count = parse_and_filter_label_file(lbl_path, class_mapping)

            total_removed_annotations += removed_count

            if len(kept_lines) > 0:
                # Copy image file to output split/images
                out_img_path = split_out_img / img_path.name
                shutil.copy2(img_path, out_img_path)

                # Write filtered labels to output split/labels
                out_lbl_path = split_out_lbl / lbl_path.name
                with open(out_lbl_path, "w", encoding="utf-8") as f:
                    f.writelines(kept_lines)

                images_copied_per_split[split] += 1
                total_kept_per_class.update(kept_counts)
            else:
                # Zero remaining annotations - do NOT copy image or create label file
                total_skipped_images += 1

    # Generate new data.yaml
    generate_data_yaml(output_dir, class_names)

    # Print Summary Report
    print("\n" + "=" * 60)
    print("DATASET FILTERING DETAILED SUMMARY")
    print("=" * 60)
    print(f"Total images copied for train: {images_copied_per_split.get('train', 0)}")
    print(f"Total images copied for valid: {images_copied_per_split.get('valid', 0)}")
    print(f"Total images copied for test:  {images_copied_per_split.get('test', 0)}")
    print("-" * 60)
    print("Total annotations kept for each class:")
    for cid in sorted(class_names.keys()):
        cname = class_names[cid]
        count = total_kept_per_class.get(cid, 0)
        print(f"  Class {cid} ({cname}): {count}")
    print("-" * 60)
    print(f"Total annotations removed: {total_removed_annotations}")
    print(f"Images skipped because no selected classes remained: {total_skipped_images}")
    print("=" * 60)

    # Validate the output dataset
    validate_output_dataset(output_dir, splits, expected_nc=len(class_names))


if __name__ == "__main__":
    try:
        filter_dataset(
            input_dir=INPUT_DATASET_DIR,
            output_dir=OUTPUT_DATASET_DIR,
            splits=SPLITS,
            class_mapping=CLASS_MAPPING,
            class_names=NEW_CLASS_NAMES
        )
    except Exception as err:
        print(f"\n[FATAL ERROR] Dataset filtering script failed: {err}", file=sys.stderr)
        sys.exit(1)
