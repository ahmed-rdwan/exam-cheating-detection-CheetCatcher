# ---- scoring.py — Scoring Engine ----
# This is the brain of the cheating-detection logic.  It does NOT detect
# anything itself — it takes the per-frame signals from the detector and
# head_pose modules and turns them into a time-based decision.
#
# Key design choice:  one bad frame never triggers a false alert.
# Points accumulate over time and only when a threshold is crossed
# does an alert actually fire (and only once per crossing, thanks
# to the cooldown in reliability.py).

import time
import os
import cv2
from datetime import datetime
from config import (
    YAW_THRESHOLD_DEGREES, PITCH_THRESHOLD_DEGREES,
    LOOK_AWAY_SECONDS, LOOK_AWAY_POINTS,
    FACE_MISSING_SECONDS, FACE_MISSING_POINTS,
    PHONE_POINTS, CHEATING_THRESHOLD, SCORE_DECAY_RATE,
    SCREENSHOT_DIR,
    # Body pose scoring thresholds (NEW)
    LEAN_SECONDS, LEAN_POINTS,
    REACH_SECONDS, REACH_POINTS,
    HAND_MOVEMENT_SECONDS, HAND_MOVEMENT_POINTS,
    POSTURE_SECONDS, POSTURE_POINTS,
)


class ScoringEngine:
    """
    Maintains a per-student state dictionary and updates it every frame.
    Each student (identified by track_id) has their own independent score
    that climbs when suspicious behavior is detected and decays when
    the behavior stops.
    """

    def __init__(self):
        # Keyed by track_id → dict with scoring state for that student
        self.students = {}

    def _ensure_student(self, track_id):
        """
        Create the state dict for a student the first time we see them.
        This avoids KeyError and keeps initialization in one place.
        """
        if track_id not in self.students:
            self.students[track_id] = {
                "score": 0.0,
                # ---- Original behavior timers ----
                "looking_away_since": None,    # timestamp when they started looking away
                "face_missing_since": None,     # timestamp when their face disappeared
                "phone_frames": 0,              # consecutive frames with phone detected
                "look_away_awarded": False,      # True if points already given for this look-away episode
                "face_missing_awarded": False,   # True if points already given for this face-missing episode
                # ---- Body pose behavior timers (NEW) ----
                "leaning_since": None,           # timestamp when leaning started
                "leaning_awarded": False,
                "reaching_since": None,          # timestamp when reaching started
                "reaching_awarded": False,
                "hand_movement_since": None,     # timestamp when excessive hand movement started
                "hand_movement_awarded": False,
                "posture_deviation_since": None,  # timestamp when posture went off-normal
                "posture_deviation_awarded": False,
                "prev_keypoints": None,           # previous frame's keypoints for velocity calc
                # ---- Alert state ----
                "alerted": False,                # True if an alert has already fired (reset by cooldown)
                "last_alert_time": 0,            # timestamp of the last fired alert
                "behaviors": [],                 # list of behavior strings active right now
            }

    def update(self, track_id, has_phone, head_pose,
              is_leaning=False, is_reaching=False,
              is_hand_moving=False, is_posture_off=False,
              current_time=None):
        """
        Update scoring for one student based on this frame's signals.

        Args:
            track_id:       int — the BoT-SORT tracker ID for this student
            has_phone:      bool — whether a phone was associated to this student
            head_pose:      dict — {'yaw', 'pitch', 'roll', 'face_detected'}
            is_leaning:     bool — leaning toward a neighbor (from body_pose)
            is_reaching:    bool — reaching beyond own box (from body_pose)
            is_hand_moving: bool — excessive hand movement (from body_pose)
            is_posture_off: bool — off normal writing posture (from body_pose)
            current_time:   float — time.time(), injectable for testing

        Returns:
            dict with keys:
              'score'      — current score for this student
              'alert'      — True if the threshold was just crossed NOW
              'behaviors'  — list of active behavior strings (for display)
        """
        if current_time is None:
            current_time = time.time()

        self._ensure_student(track_id)
        state = self.students[track_id]
        state["behaviors"] = []

        # ---- Rule 1: Phone detection ----
        # Phone is an instant high-severity signal: if the reliability
        # layer confirms it (handled externally), add full points immediately.
        if has_phone:
            state["score"] += PHONE_POINTS
            state["behaviors"].append("Phone Detected")

        # ---- Rule 2: Looking away ----
        # We check if the yaw (left/right) exceeds the threshold.
        # Points are added only after the student has been looking away
        # for LOOK_AWAY_SECONDS continuously.
        face_detected = head_pose.get("face_detected", False)
        yaw = abs(head_pose.get("yaw", 0))

        if face_detected and yaw > YAW_THRESHOLD_DEGREES:
            # Student is currently looking away
            if state["looking_away_since"] is None:
                # Start the timer — first frame of this look-away episode
                state["looking_away_since"] = current_time

            elapsed = current_time - state["looking_away_since"]
            state["behaviors"].append(f"Looking Away ({elapsed:.1f}s)")

            if elapsed >= LOOK_AWAY_SECONDS and not state["look_away_awarded"]:
                # Timer expired — this is a sustained look-away, add points
                state["score"] += LOOK_AWAY_POINTS
                state["look_away_awarded"] = True
        else:
            # Student is facing forward (or no face) — reset look-away timer
            state["looking_away_since"] = None
            state["look_away_awarded"] = False

        # ---- Rule 3: Face missing ----
        # If no face is detected inside the student's bounding box,
        # they might be hiding, turning completely around, or gone.
        if not face_detected:
            if state["face_missing_since"] is None:
                state["face_missing_since"] = current_time

            elapsed = current_time - state["face_missing_since"]
            state["behaviors"].append(f"Face Missing ({elapsed:.1f}s)")

            if elapsed >= FACE_MISSING_SECONDS and not state["face_missing_awarded"]:
                state["score"] += FACE_MISSING_POINTS
                state["face_missing_awarded"] = True
        else:
            # Face is visible again — reset the timer
            state["face_missing_since"] = None
            state["face_missing_awarded"] = False

        # ---- Rule 4: Leaning toward neighbor (NEW — body pose) ----
        if is_leaning:
            if state["leaning_since"] is None:
                state["leaning_since"] = current_time
            elapsed = current_time - state["leaning_since"]
            state["behaviors"].append(f"Leaning ({elapsed:.1f}s)")
            if elapsed >= LEAN_SECONDS and not state["leaning_awarded"]:
                state["score"] += LEAN_POINTS
                state["leaning_awarded"] = True
        else:
            state["leaning_since"] = None
            state["leaning_awarded"] = False

        # ---- Rule 5: Reaching toward others (NEW — body pose) ----
        if is_reaching:
            if state["reaching_since"] is None:
                state["reaching_since"] = current_time
            elapsed = current_time - state["reaching_since"]
            state["behaviors"].append(f"Reaching ({elapsed:.1f}s)")
            if elapsed >= REACH_SECONDS and not state["reaching_awarded"]:
                state["score"] += REACH_POINTS
                state["reaching_awarded"] = True
        else:
            state["reaching_since"] = None
            state["reaching_awarded"] = False

        # ---- Rule 6: Excessive hand movement (NEW — body pose) ----
        if is_hand_moving:
            if state["hand_movement_since"] is None:
                state["hand_movement_since"] = current_time
            elapsed = current_time - state["hand_movement_since"]
            state["behaviors"].append(f"Hand Movement ({elapsed:.1f}s)")
            if elapsed >= HAND_MOVEMENT_SECONDS and not state["hand_movement_awarded"]:
                state["score"] += HAND_MOVEMENT_POINTS
                state["hand_movement_awarded"] = True
        else:
            state["hand_movement_since"] = None
            state["hand_movement_awarded"] = False

        # ---- Rule 7: Posture deviation (NEW — body pose) ----
        if is_posture_off:
            if state["posture_deviation_since"] is None:
                state["posture_deviation_since"] = current_time
            elapsed = current_time - state["posture_deviation_since"]
            state["behaviors"].append(f"Off Posture ({elapsed:.1f}s)")
            if elapsed >= POSTURE_SECONDS and not state["posture_deviation_awarded"]:
                state["score"] += POSTURE_POINTS
                state["posture_deviation_awarded"] = True
        else:
            state["posture_deviation_since"] = None
            state["posture_deviation_awarded"] = False

        # ---- Score decay ----
        # When no suspicious behavior is happening, the score slowly
        # drops back toward zero.  This prevents a single past incident
        # from keeping a student flagged forever.
        if not state["behaviors"]:
            decay = SCORE_DECAY_RATE  # points per update cycle
            state["score"] = max(0.0, state["score"] - decay)

        # ---- Threshold check ----
        # Only fire an alert the moment the score crosses the threshold,
        # not every frame it stays above.  The 'alerted' flag prevents
        # repeated alerts — the cooldown logic in reliability.py resets it.
        alert_now = False
        if state["score"] >= CHEATING_THRESHOLD and not state["alerted"]:
            alert_now = True
            state["alerted"] = True
            state["last_alert_time"] = current_time

        return {
            "score": state["score"],
            "alert": alert_now,
            "behaviors": list(state["behaviors"]),
        }

    def reset_alert_flag(self, track_id):
        """
        Called by the reliability layer after the cooldown period expires,
        allowing the same student to trigger another alert if the behavior
        continues or reoccurs.
        """
        if track_id in self.students:
            self.students[track_id]["alerted"] = False

    def get_score(self, track_id):
        """Return the current score for a student, or 0 if unknown."""
        if track_id in self.students:
            return self.students[track_id]["score"]
        return 0.0

    def cleanup_stale(self, active_track_ids):
        """
        Remove state for students who are no longer being tracked
        (e.g. they left the frame).  This prevents the dict from
        growing unboundedly during a long video.
        """
        stale = [tid for tid in self.students if tid not in active_track_ids]
        for tid in stale:
            del self.students[tid]


# ---- Screenshot Capture (Phase 9) ----

def capture_screenshot(frame, track_id, person_box=None):
    """
    Save a screenshot when an alert fires — this is the visual evidence.

    Saves the full frame (with the student's region clearly visible)
    to alerts/screenshots/ with a descriptive filename.

    Args:
        frame:      the current video frame (full, not cropped)
        track_id:   which student triggered the alert
        person_box: optional (x1,y1,x2,y2) — if given, also saves a cropped version

    Returns:
        str — the file path where the screenshot was saved
    """
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"track{track_id}_{timestamp}.jpg"
    filepath = os.path.join(SCREENSHOT_DIR, filename)

    # Save the full frame so the reviewer has context (surrounding students, etc.)
    cv2.imwrite(filepath, frame)

    return filepath
