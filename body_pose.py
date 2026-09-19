# ---- body_pose.py — Body Pose Analysis from YOLO Keypoints ----
# This module analyzes body posture using the 17 COCO keypoints
# returned by YOLO Pose Estimation.  It detects 4 new suspicious
# behaviors that can't be caught by head pose or phone detection alone:
#
#   1. Leaning toward a nearby student
#   2. Reaching across or toward others
#   3. Unusual / excessive hand movement
#   4. Being off a normal writing posture
#
# Each function returns a simple bool — the scoring engine and
# reliability layer handle the time-based and persistence logic.
#
# COCO Keypoint indices (17 total):
#   0:nose  1:left_eye  2:right_eye  3:left_ear  4:right_ear
#   5:left_shoulder  6:right_shoulder  7:left_elbow  8:right_elbow
#   9:left_wrist  10:right_wrist  11:left_hip  12:right_hip
#   13:left_knee  14:right_knee  15:left_ankle  16:right_ankle

import math
import numpy as np
from config import (
    LEAN_DISTANCE_RATIO,
    REACH_BEYOND_RATIO,
    HAND_VELOCITY_THRESHOLD,
    POSTURE_ANGLE_THRESHOLD,
)

# ---- Keypoint index constants for readability ----
KP_NOSE = 0
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12


def parse_keypoints(raw_keypoints):
    """
    Convert raw YOLO Pose keypoint array into a structured dict.

    Args:
        raw_keypoints: numpy array of shape (17, 2) or (17, 3).
                       Each row is [x, y] or [x, y, confidence].

    Returns:
        dict mapping keypoint name → (x, y) tuple, or None if
        the keypoints array is invalid/empty.
        Also includes a 'valid' key indicating if enough keypoints
        were detected (confidence > 0.3 for the key body points).
    """
    if raw_keypoints is None or len(raw_keypoints) == 0:
        return None

    kp = np.array(raw_keypoints)
    if kp.shape[0] < 17:
        return None

    # Extract confidence if available (shape is (17,3) with x,y,conf)
    has_conf = kp.shape[1] >= 3

    def get_point(idx):
        """Get (x, y) for a keypoint, return None if confidence too low."""
        if has_conf and kp[idx][2] < 0.3:
            return None
        return (float(kp[idx][0]), float(kp[idx][1]))

    result = {
        "nose": get_point(KP_NOSE),
        "left_shoulder": get_point(KP_LEFT_SHOULDER),
        "right_shoulder": get_point(KP_RIGHT_SHOULDER),
        "left_elbow": get_point(KP_LEFT_ELBOW),
        "right_elbow": get_point(KP_RIGHT_ELBOW),
        "left_wrist": get_point(KP_LEFT_WRIST),
        "right_wrist": get_point(KP_RIGHT_WRIST),
        "left_hip": get_point(KP_LEFT_HIP),
        "right_hip": get_point(KP_RIGHT_HIP),
    }

    # Check if the critical body keypoints are detected
    # (we need at least shoulders to do any meaningful analysis)
    critical = ["left_shoulder", "right_shoulder"]
    result["valid"] = all(result[k] is not None for k in critical)

    return result


def _midpoint(p1, p2):
    """Calculate the midpoint between two (x, y) points."""
    if p1 is None or p2 is None:
        return None
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def _distance(p1, p2):
    """Euclidean distance between two (x, y) points."""
    if p1 is None or p2 is None:
        return 0.0
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# ---- 1. Leaning Detection ----

def detect_leaning(keypoints, own_box, neighbor_boxes):
    """
    Check if the student is leaning toward a neighboring student.

    Strategy:
      1. Calculate the torso midpoint (average of shoulders and hips).
      2. Calculate the center of the student's own bounding box.
      3. If the torso midpoint has shifted toward any neighbor by more
         than LEAN_DISTANCE_RATIO of the student's own box width,
         the student is "leaning."

    Why this works: when someone leans sideways to look at another
    student's paper, their torso midpoint shifts noticeably toward
    the neighbor while their bounding box center stays roughly fixed
    (the box expands rather than moves).

    Args:
        keypoints:      parsed keypoints dict from parse_keypoints()
        own_box:        (x1, y1, x2, y2) — this student's bounding box
        neighbor_boxes: list of (x1, y1, x2, y2) — other students' boxes

    Returns:
        bool — True if the student is leaning toward a neighbor
    """
    if keypoints is None or not keypoints["valid"]:
        return False
    if not neighbor_boxes:
        return False

    # Calculate torso midpoint from shoulders (and hips if available)
    shoulder_mid = _midpoint(keypoints["left_shoulder"], keypoints["right_shoulder"])
    hip_mid = _midpoint(keypoints["left_hip"], keypoints["right_hip"])

    if shoulder_mid is None:
        return False

    # Use shoulder-hip midpoint if hips are visible, otherwise just shoulders
    if hip_mid is not None:
        torso_mid = _midpoint(shoulder_mid, hip_mid)
    else:
        torso_mid = shoulder_mid

    # Center of the student's own bounding box
    x1, y1, x2, y2 = own_box
    box_center_x = (x1 + x2) / 2
    box_width = x2 - x1

    # How far has the torso shifted from the box center (horizontally)?
    torso_shift = abs(torso_mid[0] - box_center_x)
    shift_ratio = torso_shift / max(box_width, 1)

    if shift_ratio < LEAN_DISTANCE_RATIO:
        return False

    # Check if the shift direction is toward any neighbor
    lean_direction = 1 if torso_mid[0] > box_center_x else -1

    for nb in neighbor_boxes:
        nb_center_x = (nb[0] + nb[2]) / 2
        neighbor_direction = 1 if nb_center_x > box_center_x else -1

        # Leaning is suspicious only if it's toward a neighbor
        if lean_direction == neighbor_direction:
            return True

    return False


# ---- 2. Reaching Detection ----

def detect_reaching(keypoints, own_box, neighbor_boxes):
    """
    Check if the student's hand extends beyond their own bounding box
    toward another student — could indicate passing notes or copying.

    Strategy:
      1. Check if either wrist is outside the student's own box.
      2. If the wrist extends beyond the box edge by more than
         REACH_BEYOND_RATIO of the box width, AND the extension
         is toward a neighbor, flag it.

    Args:
        keypoints:      parsed keypoints dict
        own_box:        (x1, y1, x2, y2)
        neighbor_boxes: list of (x1, y1, x2, y2)

    Returns:
        bool — True if the student is reaching toward another student
    """
    if keypoints is None or not keypoints["valid"]:
        return False
    if not neighbor_boxes:
        return False

    x1, y1, x2, y2 = own_box
    box_width = x2 - x1
    threshold = box_width * REACH_BEYOND_RATIO

    # Check both wrists
    for wrist_key in ["left_wrist", "right_wrist"]:
        wrist = keypoints.get(wrist_key)
        if wrist is None:
            continue

        wx, wy = wrist

        # How far does the wrist extend beyond the box in each direction?
        extend_left = x1 - wx      # positive if wrist is left of the box
        extend_right = wx - x2     # positive if wrist is right of the box

        if extend_left > threshold:
            # Wrist extends to the left — check if any neighbor is to the left
            for nb in neighbor_boxes:
                nb_center_x = (nb[0] + nb[2]) / 2
                if nb_center_x < x1:  # neighbor is to the left
                    return True

        if extend_right > threshold:
            # Wrist extends to the right — check if any neighbor is to the right
            for nb in neighbor_boxes:
                nb_center_x = (nb[0] + nb[2]) / 2
                if nb_center_x > x2:  # neighbor is to the right
                    return True

    return False


# ---- 3. Excessive Hand Movement Detection ----

def detect_hand_movement(keypoints, prev_keypoints):
    """
    Detect unusual or excessive hand movement by tracking wrist
    displacement between consecutive frames.

    Strategy:
      1. Compare current wrist positions with the previous frame's.
      2. If the average displacement exceeds HAND_VELOCITY_THRESHOLD
         pixels, flag it as "excessive movement."

    Why this matters: a student writing normally has small, consistent
    hand movements.  Someone nervously passing notes, signaling, or
    fidgeting with a hidden device shows large, erratic wrist motion.

    Args:
        keypoints:      current frame's parsed keypoints
        prev_keypoints: previous frame's parsed keypoints (None if first frame)

    Returns:
        bool — True if hand movement is excessive
    """
    if keypoints is None or prev_keypoints is None:
        return False
    if not keypoints["valid"] or not prev_keypoints["valid"]:
        return False

    total_velocity = 0.0
    wrist_count = 0

    for wrist_key in ["left_wrist", "right_wrist"]:
        curr = keypoints.get(wrist_key)
        prev = prev_keypoints.get(wrist_key)
        if curr is not None and prev is not None:
            velocity = _distance(curr, prev)
            total_velocity += velocity
            wrist_count += 1

    if wrist_count == 0:
        return False

    avg_velocity = total_velocity / wrist_count
    return avg_velocity > HAND_VELOCITY_THRESHOLD


# ---- 4. Posture Deviation Detection ----

def detect_posture_deviation(keypoints):
    """
    Check if the student's body posture deviates from a normal
    writing position — i.e., the torso is tilted sideways.

    Strategy:
      1. Draw a line from the shoulder midpoint to the hip midpoint.
      2. Calculate the angle of this line relative to vertical.
      3. If the angle exceeds POSTURE_ANGLE_THRESHOLD, the student
         is "off posture."

    Why this works: a normal seated writing posture keeps the
    shoulder-hip line roughly vertical.  Leaning far to the side,
    twisting to look at another desk, or bending under the desk
    all cause significant deviation.

    Args:
        keypoints: parsed keypoints dict

    Returns:
        bool — True if posture deviates beyond threshold
    """
    if keypoints is None or not keypoints["valid"]:
        return False

    shoulder_mid = _midpoint(keypoints["left_shoulder"], keypoints["right_shoulder"])
    hip_mid = _midpoint(keypoints["left_hip"], keypoints["right_hip"])

    if shoulder_mid is None or hip_mid is None:
        return False

    # Vector from hip midpoint to shoulder midpoint
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]

    # Angle from vertical (vertical = straight up = dy is negative in image coords)
    # atan2 gives the angle of the vector; we want deviation from vertical (90°)
    # In image coordinates, y increases downward, so a vertical torso has
    # dx ≈ 0 and dy < 0 (shoulders above hips).
    if abs(dy) < 1:  # nearly horizontal torso — definitely off posture
        return True

    angle_from_vertical = abs(math.degrees(math.atan2(dx, -dy)))
    return angle_from_vertical > POSTURE_ANGLE_THRESHOLD
