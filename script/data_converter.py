"""
data_converter.py

Part of: BatteryVisionAI
Purpose: Converts the raw Kaggle "Battery Electrode Coating Defect Dataset"
         into a clean train/val/test split for CNN classifier training
         (ResNet18 / EfficientNet-lite stage of the AOI pipeline).

--------------------------------------------------------------------------
SOURCE DATASET
--------------------------------------------------------------------------
Battery Electrode Coating Defect Dataset (Kaggle).

Expected raw layout:

    dataset/archive/classification/
        images/          *.jpg / *.png
        labels.csv       one-hot encoded columns:
                          file_name, Surface_Crack, Pinhole, Delamination, unclassified

Each row in labels.csv represents one image, with a 1 in exactly one
(or more) of the defect columns indicating which defect(s) are present.

--------------------------------------------------------------------------
LABEL MAPPING (dataset label -> project class)
--------------------------------------------------------------------------
    Surface_Crack   -> scratch        Visible breaks / hairline cracks in the coating
    Pinhole         -> dust           Microscopic punctures / air pockets in the film
    Delamination    -> stain          Active material layer peeling/lifting from the foil
    unclassified    -> unclassified   Irregular anomalies that don't fit the other 3 types
                                       (kept SEPARATE from stain deliberately -- merging it
                                       with delamination would make "stain" visually
                                       inconsistent and hurt classifier training)

--------------------------------------------------------------------------
OUTPUT
--------------------------------------------------------------------------
    dataset/converted_data/
        train/scratch/  train/dust/  train/stain/  train/unclassified/
        val/scratch/    val/dust/    val/stain/    val/unclassified/
        test/scratch/   test/dust/   test/stain/   test/unclassified/

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python data_converter.py \
        --source dataset/archive/classification \
        --dest dataset/converted_data \
        --resize 224 224 \
        --train 0.7 --val 0.15 --test 0.15

The default label mapping above is applied automatically. To override it
(e.g. for a different dataset release, or to test merging "unclassified"
into "stain"), pass --class_map explicitly:

    python data_converter.py \
        --source dataset/archive/classification \
        --dest dataset/converted_data \
        --class_map "Surface_Crack:scratch,Pinhole:dust,Delamination:stain,unclassified:stain"
"""

import argparse
import csv
import random
import shutil
from pathlib import Path

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Common column name variants seen in Kaggle CSVs (for auto-detection on
# other dataset releases / re-exports with slightly different headers).
FILENAME_KEYS = ["file_name", "filename", "original_file_name", "image", "image_name", "img", "file", "image_id"]
LABEL_KEYS = ["label", "class", "defect", "defect_type", "category", "target"]

# Default mapping for the Battery Electrode Coating Defect Dataset.
# unclassified is intentionally kept as its own class rather than merged
# into "stain" -- see docstring above.
DEFAULT_CLASS_MAP = {
    "Surface_Crack": "scratch",
    "Pinhole": "dust",
    "Delamination": "stain",
    # "unclassified" is left unmapped on purpose -> stays as its own folder
}


def detect_filename_column(fieldnames):
    fieldnames_lower = {f.lower().strip(): f for f in fieldnames}
    for key in FILENAME_KEYS:
        if key in fieldnames_lower:
            return fieldnames_lower[key]
    return None


def detect_label_column(fieldnames):
    fieldnames_lower = {f.lower().strip(): f for f in fieldnames}
    return next((fieldnames_lower[k] for k in LABEL_KEYS if k in fieldnames_lower), None)


def load_labels(csv_path: Path):
    """
    Returns dict: {class_name: [image_filename, ...]}

    Supports two CSV formats:
      1. Long format:    filename,label
      2. One-hot format: filename, ClassA, ClassB, ClassC, ...
         where each class column holds 0/1 (or True/False) and the
         column(s) with value 1 are the label(s) for that row.
    """
    grouped = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        fname_col = detect_filename_column(fieldnames)
        label_col = detect_label_column(fieldnames)

        if fname_col is None:
            raise SystemExit(
                f"Could not auto-detect a filename column from CSV headers: {fieldnames}\n"
                f"Add your actual column name to FILENAME_KEYS in this script."
            )

        if label_col is not None:
            print(f"Detected long-format CSV -> filename: '{fname_col}', label: '{label_col}'")
            for row in reader:
                fname = row[fname_col].strip()
                label = row[label_col].strip()
                if not fname or not label:
                    continue
                grouped.setdefault(label, []).append(fname)
            return grouped

        # One-hot format: every other column is a candidate class column.
        class_cols = [c for c in fieldnames if c != fname_col]
        print("No single label column found -- treating CSV as one-hot format.")
        print(f"Filename column: '{fname_col}'  |  Class columns: {class_cols}")

        multi_label_count = 0
        no_label_count = 0
        for row in reader:
            fname = row[fname_col].strip()
            if not fname:
                continue
            active = [c for c in class_cols if str(row.get(c, "")).strip() in ("1", "1.0", "True", "true")]
            if len(active) == 0:
                no_label_count += 1
                continue
            if len(active) > 1:
                multi_label_count += 1
            for label in active:
                grouped.setdefault(label, []).append(fname)

        if multi_label_count:
            print(f"  [WARN] {multi_label_count} rows had more than one class flag set to 1; "
                  f"those images were added to each matching class folder.")
        if no_label_count:
            print(f"  [WARN] {no_label_count} rows had no class flag set to 1 (all zero); skipped.")

        return grouped


def split_list(items, ratios, seed=42):
    items = items[:]
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def resolve_image_path(images_dir: Path, fname: str):
    """Handles CSVs that store filenames without an extension or with mismatched case."""
    candidate = images_dir / fname
    if candidate.exists():
        return candidate
    if not Path(fname).suffix:
        for ext in IMG_EXTS:
            c = images_dir / f"{fname}{ext}"
            if c.exists():
                return c
    lower_target = fname.lower()
    for f in images_dir.iterdir():
        if f.name.lower() == lower_target:
            return f
    return None


def copy_or_resize(src_path: Path, dst_path: Path, target_size=None):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if target_size is None:
        shutil.copy2(src_path, dst_path)
        return
    try:
        img = Image.open(src_path).convert("RGB")
        img = img.resize(target_size)
        out_path = dst_path.with_suffix(".jpg")
        img.save(out_path, quality=95)
    except Exception as e:
        print(f"  [WARN] failed to process {src_path}: {e}")


def parse_class_map(raw: str):
    mapping = {}
    if not raw:
        return mapping
    for pair in raw.split(","):
        old, new = pair.split(":")
        mapping[old.strip()] = new.strip()
    return mapping


def main():
    ap = argparse.ArgumentParser(
        description="Convert the Battery Electrode Coating Defect Dataset into a train/val/test split for CNN training."
    )
    ap.add_argument("--source", required=True, help="Path to dataset/archive/classification (must contain images/ and labels.csv)")
    ap.add_argument("--dest", required=True, help="Destination root, e.g. dataset/converted_data")
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--resize", type=int, nargs=2, default=None, metavar=("W", "H"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--class_map",
        type=str,
        default=None,
        help=(
            "Override the default label mapping, format 'old:new,old:new'. "
            "Default mapping (Battery Electrode Coating Defect Dataset): "
            f"{', '.join(f'{k}->{v}' for k, v in DEFAULT_CLASS_MAP.items())}. "
            "Any label not in the map (e.g. 'unclassified') keeps its original name."
        ),
    )
    args = ap.parse_args()

    ratios = (args.train, args.val, args.test)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise SystemExit(f"train+val+test must sum to 1.0 (got {sum(ratios)})")

    source = Path(args.source)
    images_dir = source / "images"
    csv_path = source / "labels.csv"

    if not images_dir.exists():
        raise SystemExit(f"images/ folder not found at {images_dir}")
    if not csv_path.exists():
        raise SystemExit(f"labels.csv not found at {csv_path}")

    class_map = parse_class_map(args.class_map) if args.class_map else dict(DEFAULT_CLASS_MAP)
    grouped = load_labels(csv_path)

    print(f"Found {len(grouped)} labels in CSV: {list(grouped.keys())}")
    print(f"Applying class map: {class_map}")

    dest = Path(args.dest)
    target_size = tuple(args.resize) if args.resize else None
    summary = {}
    missing = 0

    for raw_label, filenames in grouped.items():
        class_name = class_map.get(raw_label, raw_label)

        resolved = []
        for fname in filenames:
            img_path = resolve_image_path(images_dir, fname)
            if img_path is None:
                missing += 1
                continue
            resolved.append(img_path)

        if not resolved:
            print(f"  [SKIP] {raw_label}: no matching image files found")
            continue

        train_imgs, val_imgs, test_imgs = split_list(resolved, ratios, seed=args.seed)

        for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            out_dir = dest / split_name / class_name
            for img_path in split_imgs:
                dst_path = out_dir / img_path.name
                copy_or_resize(img_path, dst_path, target_size=target_size)

        summary[class_name] = summary.get(class_name, 0) + len(resolved)
        print(f"  [OK] {raw_label} -> {class_name}: {len(resolved)} images "
              f"(train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)})")

    print("\n=== Summary ===")
    total = 0
    for cls, count in summary.items():
        print(f"  {cls}: {count} images")
        total += count
    print(f"  TOTAL: {total} images")
    if missing:
        print(f"  [WARN] {missing} filenames listed in labels.csv had no matching image file")
    print(f"\nConverted dataset ready at: {dest.resolve()}")


if __name__ == "__main__":
    main()