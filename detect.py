import onnxruntime
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import sys
import os

# ── Settings ──────────────────────────────────────────────────────────────────
MODEL_PATH   = "shoe_detector_nano.onnx"
INPUT_W      = 416
INPUT_H      = 416
CONF_THRESH  = 0.25
NMS_THRESH   = 0.45
CLASS_NAMES  = ["shoe", "shoe-alt"]  # must match your training classes

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading model: {MODEL_PATH}")
session = onnxruntime.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
print("Model loaded successfully.\n")

# ── Pre-processing ─────────────────────────────────────────────────────────────
def preprocess(image_path):
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img = cv2.resize(img_orig, (INPUT_W, INPUT_H))
    img = img[:, :, ::-1].astype(np.float32)   # BGR → RGB
    img /= 255.0
    img = np.transpose(img, (2, 0, 1))          # HWC → CHW
    img = np.expand_dims(img, axis=0)            # → 1xCxHxW
    return np.ascontiguousarray(img), img_orig

# ── Post-processing ────────────────────────────────────────────────────────────
def postprocess(outputs, orig_w, orig_h):
    raw = outputs[0].reshape(-1, 5 + len(CLASS_NAMES))
    boxes, scores, class_ids = [], [], []

    for det in raw:
        obj_conf = float(det[4])
        if obj_conf < CONF_THRESH:
            continue
        cls_scores = det[5:] * obj_conf
        cls_id     = int(np.argmax(cls_scores))
        score      = float(cls_scores[cls_id])
        if score < CONF_THRESH:
            continue

        cx, cy, w, h = det[:4]
        x1 = int((cx - w / 2) * orig_w / INPUT_W)
        y1 = int((cy - h / 2) * orig_h / INPUT_H)
        x2 = int((cx + w / 2) * orig_w / INPUT_W)
        y2 = int((cy + h / 2) * orig_h / INPUT_H)

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(score)
        class_ids.append(cls_id)

    detections = []
    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, NMS_THRESH)
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                detections.append((x, y, x + w, y + h, scores[i], class_ids[i]))

    return detections

# ── Draw and display ───────────────────────────────────────────────────────────
def draw_and_show(image_path, detections, orig_img):
    orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Shoe Detector — {os.path.basename(image_path)}", fontsize=13)

    # Original
    axes[0].imshow(orig_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Prediction
    axes[1].imshow(orig_rgb)
    axes[1].set_title(f"Prediction — {len(detections)} shoe(s) detected")
    axes[1].axis("off")

    for (x1, y1, x2, y2, conf, cls_id) in detections:
        label = f"{CLASS_NAMES[cls_id]} {conf:.0%}"
        rect  = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="#00ff55", facecolor="none"
        )
        axes[1].add_patch(rect)
        axes[1].text(
            x1, max(y1 - 6, 10), label,
            color="white", fontsize=9, fontweight="bold",
            bbox=dict(facecolor="#00ff55", alpha=0.7, pad=2, edgecolor="none")
        )

    plt.tight_layout()
    plt.show()

# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary(detections):
    print("\n" + "=" * 45)
    print("         DETECTION SUMMARY")
    print("=" * 45)
    if detections:
        print(f"  Shoes detected: {len(detections)}\n")
        for i, (x1, y1, x2, y2, conf, cls_id) in enumerate(detections, 1):
            print(f"  [{i}] {CLASS_NAMES[cls_id]:<12} confidence: {conf:.1%}")
            print(f"       box: ({x1}, {y1}) → ({x2}, {y2})")
    else:
        print("  No shoes detected.")
        print("  Try lowering CONF_THRESH (e.g. 0.25) if you expected a detection.")
    print("=" * 45 + "\n")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # If an image path is passed as argument, use it
    # Otherwise open a file picker dialog
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Opening file picker — select an image...")
        root = tk.Tk()
        root.withdraw()  # hide the empty tkinter window
        image_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.webp *.bmp"),
                ("All files",   "*.*")
            ]
        )
        root.destroy()

    if not image_path:
        print("No image selected. Exiting.")
        return

    print(f"Running inference on: {image_path}")

    inp, orig_img    = preprocess(image_path)
    outputs          = session.run(None, {input_name: inp})
    orig_h, orig_w   = orig_img.shape[:2]
    detections       = postprocess(outputs, orig_w, orig_h)

    print_summary(detections)
    draw_and_show(image_path, detections, orig_img)

if __name__ == "__main__":
    main()