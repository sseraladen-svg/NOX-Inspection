import os
import cv2
import shutil
import random

# ----------------------------
# Paths
# ----------------------------
MVTEC_ROOT = "dataset/mvtec_anomaly_detection/bottle"
YOLO_ROOT = "dataset/yolo_seg"

TRAIN_IMG = os.path.join(YOLO_ROOT, "images", "train")
VAL_IMG = os.path.join(YOLO_ROOT, "images", "val")
TRAIN_LABEL = os.path.join(YOLO_ROOT, "labels", "train")
VAL_LABEL = os.path.join(YOLO_ROOT, "labels", "val")

for p in [TRAIN_IMG, VAL_IMG, TRAIN_LABEL, VAL_LABEL]:
    os.makedirs(p, exist_ok=True)

random.seed(42)

# ----------------------------
# Convert PNG mask -> YOLO polygon
# ----------------------------
def mask_to_polygon(mask_path, label_path):

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        return

    h, w = mask.shape

    _, thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    lines = []

    for cnt in contours:

        if cv2.contourArea(cnt) < 20:
            continue

        epsilon = 0.002 * cv2.arcLength(cnt, True)
        poly = cv2.approxPolyDP(cnt, epsilon, True)

        pts = []

        for p in poly:
            x, y = p[0]
            pts.append(f"{x/w:.6f}")
            pts.append(f"{y/h:.6f}")

        if len(pts) >= 6:
            lines.append("0 " + " ".join(pts))

    with open(label_path, "w") as f:
        f.write("\n".join(lines))


# ----------------------------
# Process Bottle Dataset
# ----------------------------
test_path = os.path.join(MVTEC_ROOT, "test")
gt_path = os.path.join(MVTEC_ROOT, "ground_truth")

for defect in os.listdir(test_path):

    if defect == "good":
        continue

    defect_img_dir = os.path.join(test_path, defect)
    defect_mask_dir = os.path.join(gt_path, defect)

    if not os.path.exists(defect_mask_dir):
        continue

    images = sorted(os.listdir(defect_img_dir))

    for img_name in images:

        if not img_name.endswith(".png"):
            continue

        img_path = os.path.join(defect_img_dir, img_name)

        mask_path = os.path.join(
            defect_mask_dir,
            img_name.replace(".png", "_mask.png")
        )

        if not os.path.exists(mask_path):
            continue

        split = "train" if random.random() < 0.8 else "val"

        img_dst = TRAIN_IMG if split == "train" else VAL_IMG
        lbl_dst = TRAIN_LABEL if split == "train" else VAL_LABEL

        shutil.copy(
            img_path,
            os.path.join(img_dst, img_name)
        )

        mask_to_polygon(
            mask_path,
            os.path.join(
                lbl_dst,
                img_name.replace(".png", ".txt")
            )
        )

# ----------------------------
# Create data.yaml
# ----------------------------
yaml = f"""path: {YOLO_ROOT}

train: images/train
val: images/val

names:
  0: defect
"""

with open(os.path.join(YOLO_ROOT, "data.yaml"), "w") as f:
    f.write(yaml)

print("\nDone.")