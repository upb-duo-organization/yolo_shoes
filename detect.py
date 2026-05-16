import onnxruntime
import numpy as np
import cv2
import os
import time

# ── Settings ──────────────────────────────────────────────────────────────────
MODEL_PATH  = "person_detector.onnx"
INPUT_W     = 640
INPUT_H     = 640
CONF_THRESH = 0.45
NMS_THRESH  = 0.45
BOX_COLOR   = (0, 200, 255)
TEXT_COLOR  = (255, 255, 255)

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Only detect these — change to set() to detect all 80 COCO classes
FILTER_CLASSES = {"person"}

# ── Load model ─────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Make sure person_detector.onnx is in the same folder as detect.py"
    )

print(f"Loading {MODEL_PATH}...")
session     = onnxruntime.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)
input_name  = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print("Model loaded. Press Q or ESC to quit.\n")

# ── Pre-processing ─────────────────────────────────────────────────────────────
def preprocess(frame):
    img = cv2.resize(frame, (INPUT_W, INPUT_H))
    img = img[:, :, ::-1].astype(np.float32)   # BGR → RGB
    img /= 255.0
    img = np.transpose(img, (2, 0, 1))          # HWC → CHW
    img = np.expand_dims(img, axis=0)            # → 1xCxHxW
    return np.ascontiguousarray(img)

# ── Post-processing ────────────────────────────────────────────────────────────
def postprocess(output, orig_w, orig_h):
    raw = output[0]

    if raw.ndim == 3:
        raw = raw[0]   # [84, num_boxes]
        raw = raw.T    # [num_boxes, 84]

    boxes, scores, class_ids = [], [], []

    for det in raw:
        class_scores = det[4:]
        cls_id       = int(np.argmax(class_scores))
        score        = float(class_scores[cls_id])

        if score < CONF_THRESH:
            continue

        if FILTER_CLASSES and COCO_CLASSES[cls_id] not in FILTER_CLASSES:
            continue

        cx, cy, w, h = det[:4]
        x1 = int((cx - w / 2) * orig_w / INPUT_W)
        y1 = int((cy - h / 2) * orig_h / INPUT_H)
        x2 = int((cx + w / 2) * orig_w / INPUT_W)
        y2 = int((cy + h / 2) * orig_h / INPUT_H)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

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

# ── Draw bounding boxes ────────────────────────────────────────────────────────
def draw_detections(frame, detections):
    for (x1, y1, x2, y2, conf, cls_id) in detections:
        label = f"{COCO_CLASSES[cls_id]} {conf:.0%}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(
            frame,
            (x1, y1 - th - 10), (x1 + tw + 6, y1),
            BOX_COLOR, -1
        )
        cv2.putText(
            frame, label, (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2
        )
    return frame

# ── HUD ───────────────────────────────────────────────────────────────────────
def draw_hud(frame, fps, num_detections):
    h, w = frame.shape[:2]

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    label      = f"People: {num_detections}"
    color      = BOX_COLOR if num_detections > 0 else (100, 100, 100)
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(frame, label,
                (w - tw - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(frame, "Q / ESC to quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    return frame

# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    # 0 = default webcam — change to 1 or 2 if the wrong camera opens
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        print("Try changing VideoCapture(0) to VideoCapture(1)")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Webcam opened — starting detection loop...")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        orig_h, orig_w = frame.shape[:2]

        inp        = preprocess(frame)
        outputs    = session.run([output_name], {input_name: inp})
        detections = postprocess(outputs, orig_w, orig_h)

        frame = draw_detections(frame, detections)

        curr_time = time.time()
        fps       = 1.0 / (curr_time - prev_time + 1e-9)
        prev_time = curr_time

        frame = draw_hud(frame, fps, len(detections))

        cv2.imshow("Person Detector — Webcam", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()