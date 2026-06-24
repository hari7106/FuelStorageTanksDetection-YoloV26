"""
SAHI Tank Detector v3 — Circles Only (no labels/confidence text)
"""

import cv2
import requests
import numpy as np
import base64
import sys

# ─────────────────────────────────────────────
# 1. USER SETTINGS
# ─────────────────────────────────────────────
API_KEY      = "6t3RCD7zG6anysk3AjGv"
PROJECT_ID   = "mini-zwohz/1"
IMAGE_PATH   = "many.webp"
OUTPUT_PATH  = "detected_v3.jpg"

# ─────────────────────────────────────────────
# 2. DETECTION SETTINGS
# ─────────────────────────────────────────────
SLICE_SIZE        = 640
OVERLAP           = 200
CONFIDENCE_THRESH = 0.22
NMS_IOU_THRESH    = 0.25
SCALES            = [1.0, 0.65, 1.4]
MAX_TANK_FRACTION = 0.22

# ─────────────────────────────────────────────
# 3. HELPERS
# ─────────────────────────────────────────────

def apply_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def encode_slice(img_bgr):
    _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buffer).decode('utf-8')

def query_roboflow(slice_b64, api_key, project_id, confidence_pct):
    url = (
        f"https://detect.roboflow.com/{project_id}"
        f"?api_key={api_key}&confidence={confidence_pct}"
    )
    try:
        resp = requests.post(
            url, data=slice_b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json().get("predictions", [])
    except Exception as e:
        print(f"    ⚠️  {e}")
        return []

def iou(box_a, box_b):
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax1+aw, bx1+bw); iy2 = min(ay1+ah, by1+bh)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = aw*ah + bw*bh - inter
    return inter/union if union > 0 else 0.0

def multi_class_nms(boxes, confidences, labels, iou_thresh):
    if not boxes:
        return []
    kept = []
    for cls in set(labels):
        idx = [i for i, l in enumerate(labels) if l == cls]
        order = sorted(idx, key=lambda i: confidences[i], reverse=True)
        suppressed = set()
        for i in range(len(order)):
            if order[i] in suppressed:
                continue
            kept.append(order[i])
            for j in range(i+1, len(order)):
                if order[j] not in suppressed:
                    if iou(boxes[order[i]], boxes[order[j]]) > iou_thresh:
                        suppressed.add(order[j])
    return kept

# ─────────────────────────────────────────────
# 4. MAIN PIPELINE
# ─────────────────────────────────────────────
print("🚀 Tank Detector v3 — Circles Only")

if OVERLAP >= SLICE_SIZE:
    print("❌ OVERLAP must be less than SLICE_SIZE"); sys.exit(1)

image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"❌ Could not open '{IMAGE_PATH}'"); sys.exit(1)

img_h, img_w, _ = image.shape
STRIDE       = SLICE_SIZE - OVERLAP
MAX_TANK_PX  = int(SLICE_SIZE * MAX_TANK_FRACTION)
conf_pct     = int(CONFIDENCE_THRESH * 100)

all_boxes, all_confidences, all_labels = [], [], []

print(f"🔍 Slicing {img_w}×{img_h} image...")

for y in range(0, img_h, STRIDE):
    for x in range(0, img_w, STRIDE):
        y1, y2 = y, min(y + SLICE_SIZE, img_h)
        x1, x2 = x, min(x + SLICE_SIZE, img_w)
        enhanced = apply_clahe(image[y1:y2, x1:x2])

        for scale in SCALES:
            if scale == 1.0:
                proc = enhanced
            else:
                proc = cv2.resize(enhanced,
                                  (int((x2-x1)*scale), int((y2-y1)*scale)),
                                  interpolation=cv2.INTER_LINEAR)

            for pred in query_roboflow(encode_slice(proc), API_KEY, PROJECT_ID, conf_pct):
                pw, ph = pred["width"], pred["height"]
                if pw > MAX_TANK_PX/scale or ph > MAX_TANK_PX/scale:
                    continue
                pcx = pred["x"] / scale + x1
                pcy = pred["y"] / scale + y1
                avg  = (pw + ph) / 2.0 / scale
                half = avg / 2.0
                all_boxes.append([int(pcx-half), int(pcy-half), int(avg), int(avg)])
                all_confidences.append(float(pred["confidence"]))
                all_labels.append(pred.get("class", "tank"))

print(f"🧠 {len(all_boxes)} raw detections → NMS...")
kept = multi_class_nms(all_boxes, all_confidences, all_labels, NMS_IOU_THRESH)
print(f"✅ {len(kept)} tanks after NMS")

# ─────────────────────────────────────────────
# 5. DRAW — circles only, no text at all
# ─────────────────────────────────────────────
output = image.copy()

for i in kept:
    bx, by, bsize, _ = all_boxes[i]
    cx     = bx + bsize // 2
    cy     = by + bsize // 2
    radius = max(bsize // 2, 3)

    # Outer circle
    cv2.circle(output, (cx, cy), radius,     (0, 220, 0), 2)
    # Small centre dot
    cv2.circle(output, (cx, cy), 2,           (0, 255, 0), -1)

cv2.imwrite(OUTPUT_PATH, output)
print(f"💾 Saved → '{OUTPUT_PATH}'")

cv2.namedWindow("Tank Detector v3", cv2.WINDOW_NORMAL)
cv2.imshow("Tank Detector v3", output)
cv2.waitKey(0)
cv2.destroyAllWindows()