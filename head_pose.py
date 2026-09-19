# ---- head_pose.py — Head Pose Estimation from YOLO Pose Keypoints ----
# This module estimates where a student is looking by analyzing the
# 5 face keypoints provided by YOLO Pose:
#   0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
#
# Instead of running a separate MediaPipe model, we estimate head
# direction directly from these keypoints — simpler, faster, and
# avoids the mediapipe version-compatibility headache.
#
# Approach:
#   - YAW (left/right turn): estimated from the nose position relative
#     to the midpoint of the eyes.  When facing forward, the nose is
#     centered between the eyes.  As the head turns, the nose moves
#     toward one side.
#   - PITCH (up/down tilt): estimated from the vertical distance between
#     the nose and the eye midpoint, normalized by eye separation.
#   - Face missing: if no face keypoints are detected with sufficient
#     confidence, the face is considered missing.
#
# Why this works well enough:
#   We don't need sub-degree accuracy — we only need to distinguish
#   "facing forward" from "looking 20°+ to the side," which these
#   5 keypoints handle reliably.

import math
import numpy as np

# Minimum confidence for a YOLO Pose keypoint to be considered "detected"
KP_CONFIDENCE_THRESHOLD = 0.3

# YOLO Pose face keypoint indices
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4


def get_head_pose(crop=None, keypoints=None):
    """
    Estimate head pose from YOLO Pose keypoints.

    This function accepts either:
      - keypoints: raw YOLO Pose keypoints array (17, 3) — preferred
      - crop: BGR image (for backward compatibility — returns face_missing
              since we can't extract keypoints from an image alone)

    Args:
        crop:       BGR image crop (deprecated, kept for compatibility)
        keypoints:  numpy array of shape (17, 3) from YOLO Pose

    Returns:
        dict with keys:
          'yaw'           — float, estimated left/right angle in degrees
          'pitch'         — float, estimated up/down angle in degrees
          'roll'          — float, estimated head tilt in degrees
          'face_detected' — bool, True if face keypoints are visible
    """
    no_face = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "face_detected": False}

    # If no keypoints provided, we can't estimate pose
    if keypoints is None:
        return no_face

    kp = np.array(keypoints)
    if kp.shape[0] < 5:
        return no_face

    # ---- Extract the 5 face keypoints ----
    # Each keypoint is [x, y, confidence]
    has_conf = kp.shape[1] >= 3

    def get_point(idx):
        """Return (x, y) if confidence is high enough, else None."""
        if has_conf and kp[idx][2] < KP_CONFIDENCE_THRESHOLD:
            return None
        return (float(kp[idx][0]), float(kp[idx][1]))

    nose = get_point(KP_NOSE)
    left_eye = get_point(KP_LEFT_EYE)
    right_eye = get_point(KP_RIGHT_EYE)
    left_ear = get_point(KP_LEFT_EAR)
    right_ear = get_point(KP_RIGHT_EAR)

    # ---- Minimum requirement: need nose + both eyes ----
    # Without these three points we can't estimate head direction
    if nose is None or left_eye is None or right_eye is None:
        return no_face

    # ---- Calculate YAW (left/right head turn) ----
    # Strategy: when facing the camera, the nose sits at the midpoint
    # of the two eyes horizontally.  As the head turns left/right,
    # the nose shifts toward one side.
    #
    # We normalize the nose's horizontal offset by the inter-eye distance
    # so the result is scale-invariant (works at any camera distance).
    eye_mid_x = (left_eye[0] + right_eye[0]) / 2
    eye_mid_y = (left_eye[1] + right_eye[1]) / 2
    eye_distance = math.sqrt(
        (right_eye[0] - left_eye[0]) ** 2 +
        (right_eye[1] - left_eye[1]) ** 2
    )

    if eye_distance < 1:
        return no_face  # eyes too close together — unreliable

    # Horizontal offset of nose from eye midpoint, normalized by eye distance
    nose_offset_x = (nose[0] - eye_mid_x) / eye_distance

    # Map the normalized offset to approximate degrees.
    # Empirically, a nose offset of ~0.5 (half the eye distance)
    # corresponds to roughly 30° of head turn.
    yaw = nose_offset_x * 60.0

    # ---- Enhance yaw estimate with ear visibility ----
    # When someone turns their head far to one side, the ear on
    # that side disappears from view.  This is a strong signal
    # that can refine our yaw estimate for large angles.
    if left_ear is not None and right_ear is None:
        # Right ear hidden → head turned significantly to the left
        yaw = min(yaw, -25.0) if yaw < 0 else yaw
        if abs(yaw) < 20:
            yaw = -25.0
    elif right_ear is not None and left_ear is None:
        # Left ear hidden → head turned significantly to the right
        yaw = max(yaw, 25.0) if yaw > 0 else yaw
        if abs(yaw) < 20:
            yaw = 25.0

    # ---- Calculate PITCH (up/down tilt) ----
    # Strategy: the vertical distance between the nose and the eye
    # midpoint changes as the head tilts up/down.  When looking down,
    # the nose moves further below the eyes; when looking up, it moves
    # closer to (or above) them.
    nose_offset_y = (nose[1] - eye_mid_y) / eye_distance

    # Baseline: when facing forward, nose is typically ~0.6–0.8 eye distances
    # below the eye midpoint (varies by face shape, but we only need relative changes)
    baseline_y_ratio = 0.7
    pitch_offset = nose_offset_y - baseline_y_ratio
    pitch = pitch_offset * 50.0  # approximate degree mapping

    # ---- Calculate ROLL (head tilt) ----
    # The angle of the line connecting the two eyes tells us how much
    # the head is tilted to one side.
    roll = math.degrees(math.atan2(
        right_eye[1] - left_eye[1],
        right_eye[0] - left_eye[0]
    ))

    return {
        "yaw": round(yaw, 1),
        "pitch": round(pitch, 1),
        "roll": round(roll, 1),
        "face_detected": True,
    }
