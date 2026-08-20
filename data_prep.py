from pathlib import Path
from io import BytesIO
from PIL import Image, ImageDraw
import base64
import json
import random
import shutil
import tarfile
import xml.etree.ElementTree as ET
import zlib

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Locations of your original downloads in Google Drive
# ─────────────────────────────────────────────────────────────────────────────
DRIVE_ROOT = Path("/content/drive/MyDrive/ca_sam_dataset")

EBHI_ARCHIVE = next(DRIVE_ROOT.glob("*ebhiseg*.tar"))
KVASIR_ROOT = DRIVE_ROOT / "kvasir-seg" / "Kvasir-SEG" / "Kvasir-SEG"
DDTI_ROOT = DRIVE_ROOT / "ddti_dataset"
STS_ROOT = DRIVE_ROOT / "sts_2d_raw" / "data"

# The completed small, portable experiment dataset.
OUTPUT_ROOT = Path("/content/ca_sam_fixed_200")

TASKS = ["ebhi_seg", "kvasir_seg", "ddti", "sts_2d"]
LIMITS = {
    "train": 200,
    "val": 40,
    "test": 80,
}
SEED = 2026

# ─────────────────────────────────────────────────────────────────────────────
# Generic helpers
# ─────────────────────────────────────────────────────────────────────────────
def write_manifest(task_dir, split, records):
    with (task_dir / f"{split}.jsonl").open("w") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def make_task_dir(task_name):
    task_dir = OUTPUT_ROOT / task_name

    if task_dir.exists():
        shutil.rmtree(task_dir)

    (task_dir / "images").mkdir(parents=True)
    (task_dir / "masks").mkdir(parents=True)

    return task_dir


def select_records(records, task_name, split, limit):
    records = list(records)
    random.Random(f"{SEED}-{task_name}-{split}").shuffle(records)
    return records[:min(limit, len(records))]


def save_selected_pairs(task_dir, records, source_image, source_mask):
    final_records = []

    for index, record in enumerate(records):
        image_src = source_image(record)
        mask_src = source_mask(record)

        # Keep portable, normalized names regardless of source filenames.
        name = f"{index:05d}.png"

        image = Image.open(image_src).convert("RGB")
        mask = Image.open(mask_src).convert("L")

        image.save(task_dir / "images" / name)
        mask.save(task_dir / "masks" / name)

        final_records.append({
            "image": f"images/{name}",
            "mask": f"masks/{name}",
        })

    return final_records


# ─────────────────────────────────────────────────────────────────────────────
# 1. EBHI-SEG: extract archive, decode Supervisely bitmap JSON masks
# ─────────────────────────────────────────────────────────────────────────────
print("\nPreparing EBHI-SEG...")

EBHI_TEMP = Path("/content/ebhi_raw")
if EBHI_TEMP.exists():
    shutil.rmtree(EBHI_TEMP)

with tarfile.open(EBHI_ARCHIVE) as archive:
    archive.extractall(EBHI_TEMP)

EBHI_IMAGES = EBHI_TEMP / "ds" / "img"
EBHI_ANNOTATIONS = EBHI_TEMP / "ds" / "ann"

if not EBHI_IMAGES.exists() or not EBHI_ANNOTATIONS.exists():
    raise FileNotFoundError("Could not find EBHI ds/img and ds/ann after extraction.")

ebhi_pairs = []

for annotation_path in sorted(EBHI_ANNOTATIONS.glob("*.png.json")):
    image_name = annotation_path.name.removesuffix(".json")
    image_path = EBHI_IMAGES / image_name

    if not image_path.exists():
        continue

    annotation = json.loads(annotation_path.read_text())
    height = annotation["size"]["height"]
    width = annotation["size"]["width"]

    canvas = np.zeros((height, width), dtype=np.uint8)

    for obj in annotation.get("objects", []):
        bitmap = obj.get("bitmap")

        if bitmap is None:
            continue

        # Supervisely bitmap: base64 -> zlib -> PNG patch.
        patch_bytes = zlib.decompress(base64.b64decode(bitmap["data"]))
        patch = np.asarray(
            Image.open(BytesIO(patch_bytes)).convert("L")
        )
        patch = (patch > 0).astype(np.uint8) * 255

        x, y = bitmap["origin"]
        patch_h, patch_w = patch.shape

        x_end = min(x + patch_w, width)
        y_end = min(y + patch_h, height)

        if x < width and y < height and x_end > x and y_end > y:
            canvas[y:y_end, x:x_end] = np.maximum(
                canvas[y:y_end, x:x_end],
                patch[:y_end-y, :x_end-x],
            )

    # Exclude the two samples that have no foreground annotation.
    if canvas.max() == 0:
        continue

    ebhi_pairs.append({
        "image_path": image_path,
        "mask_array": canvas,
        "id": image_name,
    })

random.Random(f"{SEED}-ebhi-full").shuffle(ebhi_pairs)

ebhi_train_end = int(len(ebhi_pairs) * 0.70)
ebhi_val_end = ebhi_train_end + int(len(ebhi_pairs) * 0.15)

ebhi_full_splits = {
    "train": ebhi_pairs[:ebhi_train_end],
    "val": ebhi_pairs[ebhi_train_end:ebhi_val_end],
    "test": ebhi_pairs[ebhi_val_end:],
}

ebhi_dir = make_task_dir("ebhi_seg")

for split, records in ebhi_full_splits.items():
    selected = select_records(records, "ebhi_seg", split, LIMITS[split])
    portable_records = []

    for index, item in enumerate(selected):
        name = f"{split}_{index:05d}.png"

        Image.open(item["image_path"]).convert("RGB").save(
            ebhi_dir / "images" / name
        )
        Image.fromarray(item["mask_array"]).save(
            ebhi_dir / "masks" / name
        )

        portable_records.append({
            "image": f"images/{name}",
            "mask": f"masks/{name}",
        })

    write_manifest(ebhi_dir, split, portable_records)
    print(f"EBHI-SEG {split}: {len(portable_records)}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Kvasir-SEG: images and same-name masks
# ─────────────────────────────────────────────────────────────────────────────
print("\nPreparing Kvasir-SEG...")

kvasir_pairs = []

for image_path in sorted((KVASIR_ROOT / "images").glob("*")):
    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue

    mask_path = KVASIR_ROOT / "masks" / image_path.name

    if mask_path.exists():
        kvasir_pairs.append({
            "image_path": image_path,
            "mask_path": mask_path,
        })

random.Random(f"{SEED}-kvasir-full").shuffle(kvasir_pairs)

kvasir_train_end = int(len(kvasir_pairs) * 0.70)
kvasir_val_end = kvasir_train_end + int(len(kvasir_pairs) * 0.15)

kvasir_full_splits = {
    "train": kvasir_pairs[:kvasir_train_end],
    "val": kvasir_pairs[kvasir_train_end:kvasir_val_end],
    "test": kvasir_pairs[kvasir_val_end:],
}

kvasir_dir = make_task_dir("kvasir_seg")

for split, records in kvasir_full_splits.items():
    selected = select_records(records, "kvasir_seg", split, LIMITS[split])

    portable_records = save_selected_pairs(
        kvasir_dir,
        selected,
        source_image=lambda item: item["image_path"],
        source_mask=lambda item: item["mask_path"],
    )

    write_manifest(kvasir_dir, split, portable_records)
    print(f"Kvasir-SEG {split}: {len(portable_records)}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. DDTI: XML freehand polygons -> PNG masks, split by case ID
# ─────────────────────────────────────────────────────────────────────────────
print("\nPreparing DDTI...")

ddti_groups = {}

for xml_path in sorted(DDTI_ROOT.glob("*.xml")):
    case_id = xml_path.stem
    root = ET.parse(xml_path).getroot()

    for mark in root.findall("mark"):
        image_index = mark.findtext("image")
        svg_text = mark.findtext("svg")

        if not image_index or not svg_text:
            continue

        image_path = DDTI_ROOT / f"{case_id}_{image_index}.jpg"

        if not image_path.exists():
            continue

        try:
            polygons = json.loads(svg_text)
        except json.JSONDecodeError:
            continue

        image = Image.open(image_path).convert("RGB")
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)

        for polygon in polygons:
            points = polygon.get("points", [])

            if len(points) >= 3:
                draw.polygon(
                    [(point["x"], point["y"]) for point in points],
                    fill=255,
                )

        if mask.getbbox() is None:
            continue

        ddti_groups.setdefault(case_id, []).append({
            "image_path": image_path,
            "mask": mask.copy(),
        })

case_ids = list(ddti_groups)
random.Random(f"{SEED}-ddti-cases").shuffle(case_ids)

ddti_train_end = int(len(case_ids) * 0.70)
ddti_val_end = ddti_train_end + int(len(case_ids) * 0.15)

ddti_case_splits = {
    "train": case_ids[:ddti_train_end],
    "val": case_ids[ddti_train_end:ddti_val_end],
    "test": case_ids[ddti_val_end:],
}

ddti_dir = make_task_dir("ddti")

for split, chosen_cases in ddti_case_splits.items():
    records = [
        item
        for case_id in chosen_cases
        for item in ddti_groups[case_id]
    ]

    selected = select_records(records, "ddti", split, LIMITS[split])
    portable_records = []

    for index, item in enumerate(selected):
        name = f"{split}_{index:05d}.png"

        Image.open(item["image_path"]).convert("RGB").save(
            ddti_dir / "images" / name
        )
        item["mask"].save(ddti_dir / "masks" / name)

        portable_records.append({
            "image": f"images/{name}",
            "mask": f"masks/{name}",
        })

    write_manifest(ddti_dir, split, portable_records)
    print(f"DDTI {split}: {len(portable_records)}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. STS-2D: labeled Parquet records -> PNG images/masks
# ─────────────────────────────────────────────────────────────────────────────
print("\nPreparing STS-2D...")

sts_files = [
    STS_ROOT / "a_pxi_labeled-00000-of-00001.parquet",
    STS_ROOT / "c_pxi_labeled-00000-of-00001.parquet",
]

for file_path in sts_files:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing STS-2D file: {file_path}")

sts_groups = {}

for parquet_path in sts_files:
    dataframe = pd.read_parquet(parquet_path)

    for _, row in dataframe.iterrows():
        if not row["labeled"] or row["mask"] is None:
            continue

        image_bytes = row["image"]["bytes"]
        mask_bytes = row["mask"]["bytes"]

        if image_bytes is None or mask_bytes is None:
            continue

        subset = str(row["subset"])

        sts_groups.setdefault(subset, []).append({
            "image_bytes": image_bytes,
            "mask_bytes": mask_bytes,
        })

# Stratified 70/15/15 split across adult and child subsets.
sts_full_splits = {"train": [], "val": [], "test": []}

for subset, records in sts_groups.items():
    random.Random(f"{SEED}-sts-{subset}").shuffle(records)

    train_end = int(len(records) * 0.70)
    val_end = train_end + int(len(records) * 0.15)

    sts_full_splits["train"].extend(records[:train_end])
    sts_full_splits["val"].extend(records[train_end:val_end])
    sts_full_splits["test"].extend(records[val_end:])

sts_dir = make_task_dir("sts_2d")

for split, records in sts_full_splits.items():
    selected = select_records(records, "sts_2d", split, LIMITS[split])
    portable_records = []

    for index, item in enumerate(selected):
        name = f"{split}_{index:05d}.png"

        Image.open(BytesIO(item["image_bytes"])).convert("RGB").save(
            sts_dir / "images" / name
        )
        Image.open(BytesIO(item["mask_bytes"])).convert("L").save(
            sts_dir / "masks" / name
        )

        portable_records.append({
            "image": f"images/{name}",
            "mask": f"masks/{name}",
        })

    write_manifest(sts_dir, split, portable_records)
    print(f"STS-2D {split}: {len(portable_records)}")


# ─────────────────────────────────────────────────────────────────────────────
# Final verification
# ─────────────────────────────────────────────────────────────────────────────
print("\nFinal manifest counts")

for task in TASKS:
    task_dir = OUTPUT_ROOT / task
    counts = {}

    for split in ("train", "val", "test"):
        counts[split] = len([
            line
            for line in (task_dir / f"{split}.jsonl").read_text().splitlines()
            if line.strip()
        ])

    print(
        f"{task:<12} "
        f"train={counts['train']:>3}  "
        f"val={counts['val']:>3}  "
        f"test={counts['test']:>3}"
    )

print(f"\nReady for training: {OUTPUT_ROOT}")
