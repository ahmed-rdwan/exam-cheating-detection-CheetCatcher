# ---- detector.py — YOLO Detection + Pose Estimation + Tracking + Phone Association ----
# This module handles all computer-vision detection:
#   1. Loading two pretrained YOLO models:
#      - YOLO Pose (yolo11n-pose.pt) for person detection + body keypoints + tracking
#      - YOLO Detection (yolov8n.pt) for phone detection (class 67)
#   2. BoT-SORT tracking for stable per-student IDs across frames
#   3. Extracting 17 COCO body keypoints per tracked person
#   4. Associating detected phones to the nearest tracked student
#
# Why two models?  YOLO Pose only detects persons (with keypoints),
# it can't detect other classes like phones.  So we use the detection
# model as a lightweight secondary pass just for phones.
import torch
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

from ultralytics import YOLO
import numpy as np
from config import (
    MODEL_PATH, POSE_MODEL_PATH, CONF_THRESHOLD, PHONE_CONF_THRESHOLD,
    PERSON_CLASS, PHONE_CLASS,
    TRACKER_CONFIG, PHONE_EXPAND_RATIO,
    POSE_IMGSZ, PHONE_IMGSZ,
)


# ---- Model Loading ----

def load_model(model_path=MODEL_PATH):
    """
    Load a pretrained YOLO detection model (for phone detection).
    On the very first run, Ultralytics automatically downloads
    the weights — no manual download needed.
    """
    model = YOLO(model_path)
    return model


def load_pose_model(model_path=POSE_MODEL_PATH):
    """
    Load a pretrained YOLO Pose model (for person detection + body keypoints).
    This model returns 17 COCO keypoints per detected person,
    which we use for body posture analysis.
    """
    model = YOLO(model_path)
    return model


# ---- Pose Detection + Tracking (primary model) ----

def detect_poses(pose_model, frame):
    """
    Run YOLO Pose detection + BoT-SORT tracking on a single frame.

    Returns a list of person dicts, each with:
      'track_id'   — stable int ID across frames
      'box'        — (x1, y1, x2, y2) bounding box
      'conf'       — detection confidence
      'keypoints'  — numpy array of shape (17, 3) with [x, y, conf] per keypoint

    The 17 COCO keypoints are:
      0:nose  1:left_eye  2:right_eye  3:left_ear  4:right_ear
      5:left_shoulder  6:right_shoulder  7:left_elbow  8:right_elbow
      9:left_wrist  10:right_wrist  11:left_hip  12:right_hip
      13:left_knee  14:right_knee  15:left_ankle  16:right_ankle
    """
    # Run pose estimation with tracking.
    # persist=True keeps tracker state across frames for stable IDs.
    results = pose_model.track(
        frame,
        conf=CONF_THRESHOLD,
        imgsz=POSE_IMGSZ,
        persist=True,
        tracker=TRACKER_CONFIG,
        device=DEVICE,       # <-- استخدام كارت الشاشة
        verbose=False,
    )

    persons = []

    boxes = results[0].boxes
    keypoints_data = results[0].keypoints  # YOLO Pose-specific output

    if boxes is None or len(boxes) == 0:
        return persons

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()

    track_ids = boxes.id
    if track_ids is not None:
        track_ids = track_ids.cpu().numpy().astype(int)

    # Extract keypoints — shape is (N, 17, 3) where 3 = [x, y, confidence]
    if keypoints_data is not None and keypoints_data.data is not None:
        kp_array = keypoints_data.data.cpu().numpy()  # (N, 17, 3)
    else:
        kp_array = None

    for i in range(len(xyxy)):
        tid = int(track_ids[i]) if track_ids is not None else None
        if tid is None:
            continue  # skip untracked detections

        box = tuple(map(int, xyxy[i]))
        conf = float(confs[i])

        # Get this person's 17 keypoints (or None if unavailable)
        kps = kp_array[i] if kp_array is not None and i < len(kp_array) else None

        persons.append({
            "track_id": tid,
            "box": box,
            "conf": conf,
            "keypoints": kps,
        })

    return persons


# ---- Phone Detection (secondary model — detection only) ----

def detect_phones(detection_model, frame):
    """
    Run YOLO detection for phones ONLY (class 67).

    This is a lightweight secondary pass — we only look for phones,
    not persons (persons come from the pose model).  Using predict()
    instead of track() since we don't need to track phones across frames.

    Returns a list of phone dicts with 'box' and 'conf'.
    """
    results = detection_model.predict(
        frame,
        classes=[PHONE_CLASS],
        conf=PHONE_CONF_THRESHOLD,
        imgsz=PHONE_IMGSZ,
        device=DEVICE,       # <-- استخدام كارت الشاشة
        verbose=False,
    )

    phones = []
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return phones

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()

    for i in range(len(xyxy)):
        phones.append({
            "box": tuple(map(int, xyxy[i])),
            "conf": float(confs[i]),
        })

    return phones


# ---- Legacy function (kept for backward compatibility) ----

def detect_and_track(model, frame):
    """
    Original combined detection + tracking function.
    Still used internally if only the detection model is available.
    """
    results = model.track(
        frame,
        classes=[PERSON_CLASS, PHONE_CLASS],
        conf=CONF_THRESHOLD,
        persist=True,
        tracker=TRACKER_CONFIG,
        verbose=False,
    )

    persons = []
    phones = []

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return persons, phones

    xyxy = boxes.xyxy.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()

    track_ids = boxes.id
    if track_ids is not None:
        track_ids = track_ids.cpu().numpy().astype(int)

    for i in range(len(xyxy)):
        box = tuple(map(int, xyxy[i]))
        cls = classes[i]
        conf = float(confs[i])

        if cls == PERSON_CLASS:
            tid = int(track_ids[i]) if track_ids is not None else None
            if tid is not None:
                persons.append({
                    "track_id": tid,
                    "box": box,
                    "conf": conf,
                })
        elif cls == PHONE_CLASS:
            phones.append({
                "box": box,
                "conf": conf,
            })

    return persons, phones


# ---- Phone-to-Student Association ----

def expand_box(box, ratio=PHONE_EXPAND_RATIO):
    """
    Expand a bounding box by a percentage on each side.
    This gives a margin of error when checking whether a phone
    is close enough to a student to be "theirs."
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return (
        int(x1 - w * ratio),
        int(y1 - h * ratio),
        int(x2 + w * ratio),
        int(y2 + h * ratio),
    )


def point_in_box(px, py, box):
    """Check whether a single point (px, py) falls inside a box."""
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def associate_phones_to_students(persons, phones):
    """
    For every detected phone, figure out which tracked student
    it belongs to.  Strategy:
      1. Compute the center point of each phone box.
      2. For each student, expand their box by PHONE_EXPAND_RATIO.
      3. If the phone center falls inside the expanded student box,
         mark that student as 'phone_detected = True'.

    Returns a set of track_ids that currently have a phone near them.
    """
    phone_holders = set()

    for phone in phones:
        px1, py1, px2, py2 = phone["box"]
        phone_cx = (px1 + px2) // 2
        phone_cy = (py1 + py2) // 2

        for person in persons:
            expanded = expand_box(person["box"])
            if point_in_box(phone_cx, phone_cy, expanded):
                phone_holders.add(person["track_id"])
                break

    return phone_holders
