# ---- reliability.py — Reliability Layer (Phase 8: Professional Touches) ----
# This module is what separates a demo script from production-grade software.
# It contains three independent mechanisms that dramatically reduce false alerts:
#
#   1. Frame-persistence confirmation — a detection must be consistent across
#      multiple consecutive frames before it "counts."
#   2. Background-person filtering — ignores people walking behind the exam
#      who aren't actually seated students.
#   3. Alert cooldown — prevents the same student from spamming alerts
#      for a single ongoing incident.

import time
from config import (
    CONFIRM_FRAMES,
    RELATIVE_MIN_HEIGHT_RATIO,
    ALERT_COOLDOWN_SECONDS,
)


# ---- 1. Frame-Persistence Confirmation ----

class FramePersistence:
    """
    Requires a behavior signal (e.g. "phone detected near student #3")
    to be true for CONFIRM_FRAMES consecutive frames before it's
    considered confirmed — and also requires it to be *absent* for
    the same number of frames before it's cleared.

    Why: YOLO might misread a pen as a phone for a single frame,
    or the tracker might glitch for one frame.  Requiring persistence
    filters out these one-off false positives.
    """

    def __init__(self, required_frames=CONFIRM_FRAMES):
        self.required = required_frames
        # Tracks consecutive "True" frame count per (track_id, signal_name)
        self.positive_counts = {}
        # Tracks consecutive "False" frame count per (track_id, signal_name)
        self.negative_counts = {}
        # The last confirmed state per (track_id, signal_name)
        self.confirmed_state = {}

    def update(self, track_id, signal_name, raw_value):
        """
        Feed in one frame's raw detection result for a specific signal.

        Args:
            track_id:    int — which student
            signal_name: str — e.g. "phone", "looking_away", "face_missing"
            raw_value:   bool — what the detector says this frame

        Returns:
            bool — the persistence-filtered (confirmed) value.
                   Won't flip to True until raw_value has been True
                   for `required` frames in a row, and won't flip
                   back to False until it's been False for `required` frames.
        """
        key = (track_id, signal_name)

        if raw_value:
            # Increment the positive counter, reset the negative counter
            self.positive_counts[key] = self.positive_counts.get(key, 0) + 1
            self.negative_counts[key] = 0

            # If we've seen enough consecutive positive frames, confirm it
            if self.positive_counts[key] >= self.required:
                self.confirmed_state[key] = True
        else:
            # Increment the negative counter, reset the positive counter
            self.negative_counts[key] = self.negative_counts.get(key, 0) + 1
            self.positive_counts[key] = 0

            # If we've seen enough consecutive negative frames, clear it
            if self.negative_counts[key] >= self.required:
                self.confirmed_state[key] = False

        # Return the current confirmed state (default False if never seen)
        return self.confirmed_state.get(key, False)

    def cleanup(self, active_track_ids):
        """Remove persistence state for students who left the frame."""
        stale_keys = [k for k in self.positive_counts if k[0] not in active_track_ids]
        for k in stale_keys:
            self.positive_counts.pop(k, None)
            self.negative_counts.pop(k, None)
            self.confirmed_state.pop(k, None)


# ---- 2. Background-Person Filtering ----

def filter_background_persons(persons):
    """
    Remove detected persons who are likely in the background
    (walking behind the exam, visible through a window, etc.)
    rather than seated students.

    Strategy: find the tallest person in the frame (assumed to be
    a real seated student at a normal distance), then discard
    anyone whose box height is below RELATIVE_MIN_HEIGHT_RATIO
    of that tallest height.

    Why this works: background people are further from the camera,
    so their bounding boxes are much smaller.  A threshold of 0.6
    means anyone less than 60% of the tallest person's height
    is filtered out.
    """
    if not persons:
        return persons

    # Find the tallest bounding box (greatest y2 - y1)
    max_height = max(p["box"][3] - p["box"][1] for p in persons)

    # Keep only persons tall enough relative to the tallest
    min_height = max_height * RELATIVE_MIN_HEIGHT_RATIO
    filtered = [
        p for p in persons
        if (p["box"][3] - p["box"][1]) >= min_height
    ]

    return filtered


# ---- 3. Alert Cooldown ----

class AlertCooldown:
    """
    After a student triggers an alert, block further alerts for that
    student for ALERT_COOLDOWN_SECONDS.  This prevents a single
    ongoing incident (e.g. student still holding phone) from
    generating 10 alerts in 2 seconds.
    """

    def __init__(self, cooldown_seconds=ALERT_COOLDOWN_SECONDS):
        self.cooldown = cooldown_seconds
        # track_id → timestamp of last fired alert
        self.last_alert_times = {}

    def can_alert(self, track_id, current_time=None):
        """Check whether enough time has passed since this student's last alert."""
        if current_time is None:
            current_time = time.time()

        last_time = self.last_alert_times.get(track_id, 0)
        return (current_time - last_time) >= self.cooldown

    def record_alert(self, track_id, current_time=None):
        """Record that an alert just fired for this student."""
        if current_time is None:
            current_time = time.time()
        self.last_alert_times[track_id] = current_time

    def cleanup(self, active_track_ids):
        """Remove cooldown state for students who left the frame."""
        stale = [tid for tid in self.last_alert_times if tid not in active_track_ids]
        for tid in stale:
            del self.last_alert_times[tid]
