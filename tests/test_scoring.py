# ---- tests/test_scoring.py — Unit Tests for Core Logic (Phase 17) ----
# These tests cover the pure-logic functions that don't need a camera
# or a real video:  the scoring engine, bbox association, and reliability
# filters.  Aim: prove the core logic is correct.

import pytest
import time
import sys
import os
import numpy as np

# Add project root to path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scoring import ScoringEngine
from detector import expand_box, point_in_box, associate_phones_to_students
from reliability import FramePersistence, filter_background_persons, AlertCooldown
from body_pose import (
    parse_keypoints, detect_leaning, detect_reaching,
    detect_hand_movement, detect_posture_deviation,
)


# ============================================================
#  Scoring Engine Tests
# ============================================================

class TestScoringEngine:
    """Tests for the ScoringEngine class."""

    def setup_method(self):
        """Fresh engine for each test."""
        self.engine = ScoringEngine()

    # ---- Phone detection scoring ----

    def test_phone_adds_100_points(self):
        """A confirmed phone should instantly add PHONE_POINTS (100)."""
        result = self.engine.update(
            track_id=1,
            has_phone=True,
            head_pose={"yaw": 0, "pitch": 0, "roll": 0, "face_detected": True},
        )
        assert result["score"] >= 100, "Phone should add at least 100 points"
        assert "Phone Detected" in result["behaviors"]

    def test_no_phone_no_points(self):
        """A student with no suspicious behavior should have score 0."""
        result = self.engine.update(
            track_id=1,
            has_phone=False,
            head_pose={"yaw": 0, "pitch": 0, "roll": 0, "face_detected": True},
        )
        assert result["score"] == 0
        assert result["behaviors"] == []

    # ---- Look-away scoring ----

    def test_look_away_needs_time(self):
        """Looking away for less than LOOK_AWAY_SECONDS should NOT add points."""
        t = 1000.0
        result = self.engine.update(
            track_id=2,
            has_phone=False,
            head_pose={"yaw": 30, "pitch": 0, "roll": 0, "face_detected": True},
            current_time=t,
        )
        # Just started — no points yet (only timer started)
        assert result["score"] == 0

    def test_look_away_adds_points_after_duration(self):
        """Looking away for >= LOOK_AWAY_SECONDS should add LOOK_AWAY_POINTS."""
        t = 1000.0
        # Frame 1: start looking away
        self.engine.update(
            track_id=2, has_phone=False,
            head_pose={"yaw": 30, "pitch": 0, "roll": 0, "face_detected": True},
            current_time=t,
        )
        # Frame 2: still looking away after 5 seconds
        result = self.engine.update(
            track_id=2, has_phone=False,
            head_pose={"yaw": 30, "pitch": 0, "roll": 0, "face_detected": True},
            current_time=t + 5,
        )
        assert result["score"] >= 40, "Should have added look-away points"

    def test_look_away_resets_when_facing_forward(self):
        """Facing forward should reset the look-away timer."""
        t = 1000.0
        # Start looking away
        self.engine.update(
            track_id=3, has_phone=False,
            head_pose={"yaw": 30, "pitch": 0, "roll": 0, "face_detected": True},
            current_time=t,
        )
        # Face forward — timer should reset
        self.engine.update(
            track_id=3, has_phone=False,
            head_pose={"yaw": 5, "pitch": 0, "roll": 0, "face_detected": True},
            current_time=t + 2,
        )
        # Look away again — need full duration again
        result = self.engine.update(
            track_id=3, has_phone=False,
            head_pose={"yaw": 30, "pitch": 0, "roll": 0, "face_detected": True},
            current_time=t + 3,
        )
        assert result["score"] == 0, "Timer should have reset"

    # ---- Face missing scoring ----

    def test_face_missing_adds_points_after_duration(self):
        """Face missing for >= FACE_MISSING_SECONDS should add points."""
        t = 1000.0
        self.engine.update(
            track_id=4, has_phone=False,
            head_pose={"face_detected": False},
            current_time=t,
        )
        result = self.engine.update(
            track_id=4, has_phone=False,
            head_pose={"face_detected": False},
            current_time=t + 4,
        )
        assert result["score"] >= 50, "Should have added face-missing points"

    # ---- Threshold crossing ----

    def test_threshold_triggers_alert_once(self):
        """Crossing the cheating threshold should trigger alert only ONCE."""
        result = self.engine.update(
            track_id=5, has_phone=True,
            head_pose={"yaw": 0, "pitch": 0, "roll": 0, "face_detected": True},
        )
        assert result["alert"] is True, "First crossing should alert"

        # Same student, phone still there — should NOT alert again
        result2 = self.engine.update(
            track_id=5, has_phone=True,
            head_pose={"yaw": 0, "pitch": 0, "roll": 0, "face_detected": True},
        )
        assert result2["alert"] is False, "Should not alert twice"

    def test_reset_alert_flag_allows_re_alert(self):
        """After resetting the alert flag, the student can alert again."""
        self.engine.update(
            track_id=6, has_phone=True,
            head_pose={"yaw": 0, "pitch": 0, "roll": 0, "face_detected": True},
        )
        self.engine.reset_alert_flag(6)

        result = self.engine.update(
            track_id=6, has_phone=True,
            head_pose={"yaw": 0, "pitch": 0, "roll": 0, "face_detected": True},
        )
        assert result["alert"] is True

    # ---- Score decay ----

    def test_score_decays_when_no_behavior(self):
        """Score should decrease when no suspicious behavior is active."""
        # First: give the student some points
        self.engine.update(
            track_id=7, has_phone=True,
            head_pose={"yaw": 0, "pitch": 0, "roll": 0, "face_detected": True},
        )
        score_after_phone = self.engine.get_score(7)

        # Now: no behavior — score should decay
        self.engine.update(
            track_id=7, has_phone=False,
            head_pose={"yaw": 0, "pitch": 0, "roll": 0, "face_detected": True},
        )
        score_after_decay = self.engine.get_score(7)
        assert score_after_decay < score_after_phone

    # ---- Cleanup ----

    def test_cleanup_removes_stale_students(self):
        """Students no longer tracked should be cleaned up."""
        self.engine.update(
            track_id=10, has_phone=False,
            head_pose={"yaw": 0, "pitch": 0, "roll": 0, "face_detected": True},
        )
        assert 10 in self.engine.students
        self.engine.cleanup_stale(active_track_ids={20, 30})
        assert 10 not in self.engine.students


# ============================================================
#  Bounding Box Association Tests
# ============================================================

class TestBBoxAssociation:
    """Tests for phone-to-student bbox association."""

    def test_expand_box_grows_correctly(self):
        """expand_box should grow the box by the specified ratio."""
        box = (100, 100, 200, 200)
        expanded = expand_box(box, ratio=0.15)
        # Original width/height = 100, expansion = 15 pixels each side
        assert expanded[0] < 100  # x1 moves left
        assert expanded[1] < 100  # y1 moves up
        assert expanded[2] > 200  # x2 moves right
        assert expanded[3] > 200  # y2 moves down

    def test_point_inside_box(self):
        """A point inside the box should return True."""
        assert point_in_box(150, 150, (100, 100, 200, 200)) is True

    def test_point_outside_box(self):
        """A point outside the box should return False."""
        assert point_in_box(50, 50, (100, 100, 200, 200)) is False

    def test_point_on_edge(self):
        """A point exactly on the box edge should return True (inclusive)."""
        assert point_in_box(100, 100, (100, 100, 200, 200)) is True

    def test_associate_phone_to_nearest_student(self):
        """A phone inside a student's expanded box should be associated."""
        persons = [
            {"track_id": 1, "box": (100, 100, 200, 300), "conf": 0.9},
            {"track_id": 2, "box": (400, 100, 500, 300), "conf": 0.9},
        ]
        # Phone is right next to student 1
        phones = [{"box": (180, 200, 220, 260), "conf": 0.7}]

        holders = associate_phones_to_students(persons, phones)
        assert 1 in holders
        assert 2 not in holders

    def test_no_phone_no_association(self):
        """No phones means no associations."""
        persons = [{"track_id": 1, "box": (100, 100, 200, 300), "conf": 0.9}]
        holders = associate_phones_to_students(persons, [])
        assert len(holders) == 0


# ============================================================
#  Reliability Layer Tests
# ============================================================

class TestFramePersistence:
    """Tests for the frame-persistence filter."""

    def test_single_frame_does_not_confirm(self):
        """One positive frame should NOT confirm the signal."""
        fp = FramePersistence(required_frames=3)
        assert fp.update(1, "phone", True) is False

    def test_enough_frames_confirms(self):
        """Three consecutive positive frames should confirm."""
        fp = FramePersistence(required_frames=3)
        fp.update(1, "phone", True)
        fp.update(1, "phone", True)
        result = fp.update(1, "phone", True)
        assert result is True

    def test_interruption_resets_count(self):
        """A negative frame in the middle resets the positive counter."""
        fp = FramePersistence(required_frames=3)
        fp.update(1, "phone", True)
        fp.update(1, "phone", True)
        fp.update(1, "phone", False)  # interruption
        fp.update(1, "phone", True)
        assert fp.update(1, "phone", True) is False  # only 2 consecutive

    def test_clearing_needs_frames_too(self):
        """Once confirmed, clearing also needs required consecutive negative frames."""
        fp = FramePersistence(required_frames=2)
        fp.update(1, "phone", True)
        fp.update(1, "phone", True)  # confirmed True

        fp.update(1, "phone", False)  # 1 negative — not cleared yet
        assert fp.confirmed_state.get((1, "phone")) is True

        fp.update(1, "phone", False)  # 2 negatives — now cleared
        assert fp.confirmed_state.get((1, "phone")) is False


class TestBackgroundFiltering:
    """Tests for background-person filtering."""

    def test_filters_short_persons(self):
        """Persons much shorter than the tallest should be filtered out."""
        persons = [
            {"track_id": 1, "box": (0, 0, 100, 300), "conf": 0.9},   # height 300 (tallest)
            {"track_id": 2, "box": (200, 200, 300, 320), "conf": 0.8},  # height 120 (background)
        ]
        filtered = filter_background_persons(persons)
        assert len(filtered) == 1
        assert filtered[0]["track_id"] == 1

    def test_keeps_similar_height(self):
        """Persons of similar height should all be kept."""
        persons = [
            {"track_id": 1, "box": (0, 0, 100, 300), "conf": 0.9},   # height 300
            {"track_id": 2, "box": (200, 0, 300, 280), "conf": 0.9},  # height 280
        ]
        filtered = filter_background_persons(persons)
        assert len(filtered) == 2

    def test_empty_list(self):
        """Empty input should return empty output."""
        assert filter_background_persons([]) == []


class TestAlertCooldown:
    """Tests for the alert cooldown mechanism."""

    def test_first_alert_always_allowed(self):
        """The first alert for any student should always be allowed."""
        cd = AlertCooldown(cooldown_seconds=10)
        assert cd.can_alert(1, current_time=100) is True

    def test_cooldown_blocks_immediate_re_alert(self):
        """An alert within the cooldown period should be blocked."""
        cd = AlertCooldown(cooldown_seconds=10)
        cd.record_alert(1, current_time=100)
        assert cd.can_alert(1, current_time=105) is False

    def test_cooldown_allows_after_expiry(self):
        """An alert after the cooldown period should be allowed."""
        cd = AlertCooldown(cooldown_seconds=10)
        cd.record_alert(1, current_time=100)
        assert cd.can_alert(1, current_time=111) is True

    def test_different_students_independent(self):
        """Cooldown for one student shouldn't affect another."""
        cd = AlertCooldown(cooldown_seconds=10)
        cd.record_alert(1, current_time=100)
        assert cd.can_alert(2, current_time=101) is True


# ============================================================
#  Body Pose Analysis Tests (NEW)
# ============================================================

class TestBodyPoseLeaning:
    """Tests for leaning detection."""

    def _make_keypoints(self, shoulder_mid_x, box_center_x=150):
        """Create a minimal keypoints array with shoulder positions shifted."""
        # Shoulders positioned symmetrically around shoulder_mid_x
        kp = np.zeros((17, 3))
        kp[5] = [shoulder_mid_x - 20, 100, 0.9]   # left_shoulder
        kp[6] = [shoulder_mid_x + 20, 100, 0.9]   # right_shoulder
        kp[11] = [box_center_x - 15, 200, 0.9]     # left_hip (centered)
        kp[12] = [box_center_x + 15, 200, 0.9]     # right_hip (centered)
        return kp

    def test_leaning_toward_neighbor(self):
        """Student leaning right toward a neighbor on the right should be detected."""
        # Box is centered at x=150, width=100 (x1=100, x2=200)
        own_box = (100, 50, 200, 300)
        # Shoulders shifted far to the right (toward neighbor).
        # With shoulder_mid=220 and hips at 150, torso_mid = (185, 150),
        # shift = 35/100 = 35% > LEAN_DISTANCE_RATIO (30%).
        kp = parse_keypoints(self._make_keypoints(shoulder_mid_x=220, box_center_x=150))
        neighbor_boxes = [(250, 50, 350, 300)]  # neighbor to the right
        assert detect_leaning(kp, own_box, neighbor_boxes) is True

    def test_not_leaning_centered(self):
        """Student sitting upright should not be flagged as leaning."""
        own_box = (100, 50, 200, 300)
        kp = parse_keypoints(self._make_keypoints(shoulder_mid_x=150, box_center_x=150))
        neighbor_boxes = [(250, 50, 350, 300)]
        assert detect_leaning(kp, own_box, neighbor_boxes) is False

    def test_no_neighbors_no_leaning(self):
        """Leaning without neighbors present should not be flagged."""
        own_box = (100, 50, 200, 300)
        kp = parse_keypoints(self._make_keypoints(shoulder_mid_x=185, box_center_x=150))
        assert detect_leaning(kp, own_box, []) is False


class TestBodyPoseReaching:
    """Tests for reaching detection."""

    def _make_keypoints_with_wrist(self, wrist_x, wrist_y=150):
        """Create keypoints with right wrist at a specific position."""
        kp = np.zeros((17, 3))
        kp[5] = [140, 100, 0.9]  # left_shoulder
        kp[6] = [160, 100, 0.9]  # right_shoulder
        kp[10] = [wrist_x, wrist_y, 0.9]  # right_wrist
        return kp

    def test_reaching_beyond_box(self):
        """Wrist extending far beyond own box toward a neighbor."""
        own_box = (100, 50, 200, 300)
        kp = parse_keypoints(self._make_keypoints_with_wrist(wrist_x=260))
        neighbor_boxes = [(250, 50, 350, 300)]  # neighbor to the right
        assert detect_reaching(kp, own_box, neighbor_boxes) is True

    def test_wrist_inside_box(self):
        """Wrist staying inside own box should not be flagged."""
        own_box = (100, 50, 200, 300)
        kp = parse_keypoints(self._make_keypoints_with_wrist(wrist_x=180))
        neighbor_boxes = [(250, 50, 350, 300)]
        assert detect_reaching(kp, own_box, neighbor_boxes) is False


class TestBodyPoseHandMovement:
    """Tests for excessive hand movement detection."""

    def _make_keypoints_at(self, wrist_x, wrist_y):
        """Create keypoints with wrists at specific positions."""
        kp = np.zeros((17, 3))
        kp[5] = [140, 100, 0.9]  # left_shoulder
        kp[6] = [160, 100, 0.9]  # right_shoulder
        kp[9] = [wrist_x - 20, wrist_y, 0.9]   # left_wrist
        kp[10] = [wrist_x + 20, wrist_y, 0.9]   # right_wrist
        return kp

    def test_high_velocity_detected(self):
        """Large wrist displacement between frames should be flagged."""
        curr = parse_keypoints(self._make_keypoints_at(200, 200))
        prev = parse_keypoints(self._make_keypoints_at(100, 100))  # 141px displacement
        assert detect_hand_movement(curr, prev) is True

    def test_low_velocity_ok(self):
        """Small wrist displacement (normal writing) should not be flagged."""
        curr = parse_keypoints(self._make_keypoints_at(152, 152))
        prev = parse_keypoints(self._make_keypoints_at(150, 150))  # ~3px displacement
        assert detect_hand_movement(curr, prev) is False


class TestBodyPosePosture:
    """Tests for posture deviation detection."""

    def _make_torso_keypoints(self, shoulder_x_offset=0):
        """Create keypoints with torso tilted by moving shoulders sideways."""
        kp = np.zeros((17, 3))
        kp[5] = [140 + shoulder_x_offset, 100, 0.9]   # left_shoulder
        kp[6] = [160 + shoulder_x_offset, 100, 0.9]   # right_shoulder
        kp[11] = [140, 250, 0.9]   # left_hip (stays centered)
        kp[12] = [160, 250, 0.9]   # right_hip (stays centered)
        return kp

    def test_tilted_posture_detected(self):
        """Significantly tilted torso should be flagged."""
        kp = parse_keypoints(self._make_torso_keypoints(shoulder_x_offset=100))
        assert detect_posture_deviation(kp) is True

    def test_upright_posture_ok(self):
        """Upright posture should not be flagged."""
        kp = parse_keypoints(self._make_torso_keypoints(shoulder_x_offset=0))
        assert detect_posture_deviation(kp) is False


# ============================================================
#  Combined Scoring Tests (multiple signals)
# ============================================================

class TestCombinedScoring:
    """Tests verifying that multiple body signals combine correctly."""

    def setup_method(self):
        self.engine = ScoringEngine()

    def test_leaning_plus_hand_movement_below_threshold(self):
        """Leaning (35) + hand movement (30) = 65, still below 80 threshold."""
        t = 1000.0
        # Start both behaviors
        self.engine.update(1, False, {"yaw": 0, "pitch": 0, "roll": 0, "face_detected": True},
                          is_leaning=True, is_hand_moving=True, current_time=t)
        # Wait long enough for both timers to expire
        result = self.engine.update(1, False, {"yaw": 0, "pitch": 0, "roll": 0, "face_detected": True},
                                   is_leaning=True, is_hand_moving=True, current_time=t + 5)
        assert result["score"] >= 65
        assert result["alert"] is False  # 65 < 80

    def test_leaning_plus_look_away_triggers_alert(self):
        """Leaning (35) + looking away (40) = 75 + time accumulation → should cross 80."""
        t = 1000.0
        self.engine.update(1, False, {"yaw": 30, "pitch": 0, "roll": 0, "face_detected": True},
                          is_leaning=True, is_reaching=True, current_time=t)
        result = self.engine.update(1, False, {"yaw": 30, "pitch": 0, "roll": 0, "face_detected": True},
                                   is_leaning=True, is_reaching=True, current_time=t + 5)
        # leaning(35) + reaching(45) = 80 → alert
        assert result["alert"] is True

    def test_new_behaviors_show_in_behavior_list(self):
        """All active body-pose behaviors should appear in the behaviors list."""
        t = 1000.0
        result = self.engine.update(1, False, {"yaw": 0, "pitch": 0, "roll": 0, "face_detected": True},
                                   is_leaning=True, is_reaching=True,
                                   is_hand_moving=True, is_posture_off=True,
                                   current_time=t)
        behavior_text = " ".join(result["behaviors"])
        assert "Leaning" in behavior_text
        assert "Reaching" in behavior_text
        assert "Hand Movement" in behavior_text
        assert "Off Posture" in behavior_text


# ============================================================
#  Run with: python -m pytest tests/ -v
# ============================================================
