# ---- main.py — Tkinter GUI Entry Point (Phases 12–15) ----
# This is the main application window.  It ties together ALL the other
# modules (detector, head_pose, scoring, reliability, firebase_db)
# into a usable desktop app with:
#
#   - A video display area (shows live detection with colored boxes)
#   - Control buttons: Upload Video, Start Camera, Stop
#   - An alerts table (Treeview) showing every incident
#   - A screenshot viewer (double-click a row to see the evidence)
#   - Threading so the UI never freezes during heavy YOLO processing
#
# Threading architecture:
#   The detection/tracking/scoring loop runs in a background thread.
#   That thread puts finished frames and new alerts into a Queue.
#   The Tkinter main thread polls the queue every 15ms using root.after()
#   and updates the UI.  We NEVER touch Tkinter widgets from the
#   background thread — that would cause crashes on some platforms.

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import threading
import queue
import time
import os
from datetime import datetime

# ---- Import our project modules ----
from config import (
    QUEUE_POLL_MS, VIDEO_MAX_WIDTH, VIDEO_MAX_HEIGHT,
    APP_TITLE, YAW_THRESHOLD_DEGREES, CHEATING_THRESHOLD,
    YAW_BASELINE_FRAMES, PHONE_SKIP_FRAMES,
)

# ---- Performance: process every Nth frame for speed ----
PROCESS_EVERY_N_FRAMES = 2  # skip alternate frames — halves YOLO workload
from detector import (
    load_model, load_pose_model,
    detect_poses, detect_phones,
    associate_phones_to_students,
)
from head_pose import get_head_pose
from body_pose import (
    parse_keypoints,
    detect_leaning, detect_reaching,
    detect_hand_movement, detect_posture_deviation,
)
from scoring import ScoringEngine, capture_screenshot
from reliability import FramePersistence, filter_background_persons, AlertCooldown
from firebase_db import add_alert


# ============================================================
#  Skeleton Drawing — visualize body keypoints on the frame
# ============================================================

# COCO skeleton bone connections (pairs of keypoint indices)
_SKELETON_BONES = [
    (0, 1), (0, 2),           # nose → eyes
    (1, 3), (2, 4),           # eyes → ears
    (5, 6),                   # left shoulder → right shoulder
    (5, 7), (7, 9),           # left shoulder → elbow → wrist
    (6, 8), (8, 10),          # right shoulder → elbow → wrist
    (5, 11), (6, 12),         # shoulders → hips
    (11, 12),                 # left hip → right hip
    (11, 13), (13, 15),       # left hip → knee → ankle
    (12, 14), (14, 16),       # right hip → knee → ankle
]

# Colors for different body regions (BGR)
_BONE_COLORS = {
    "face": (255, 200, 100),       # light blue — face connections
    "arm_left": (100, 255, 100),   # green — left arm
    "arm_right": (100, 100, 255),  # red — right arm
    "torso": (255, 255, 100),      # cyan — torso
    "leg_left": (100, 255, 255),   # yellow — left leg
    "leg_right": (200, 100, 255),  # purple — right leg
}

def _bone_region(i, j):
    """Determine which body region a bone belongs to for coloring."""
    pair = (min(i, j), max(i, j))
    if pair[1] <= 4:
        return "face"
    if pair in [(5, 7), (7, 9)]:
        return "arm_left"
    if pair in [(6, 8), (8, 10)]:
        return "arm_right"
    if pair in [(5, 6), (5, 11), (6, 12), (11, 12)]:
        return "torso"
    if pair in [(11, 13), (13, 15)]:
        return "leg_left"
    if pair in [(12, 14), (14, 16)]:
        return "leg_right"
    return "torso"


def draw_skeleton(frame, keypoints_raw, conf_threshold=0.3):
    """
    Draw body keypoints and skeleton bones on the frame.

    Args:
        frame:          the BGR image to draw on (modified in-place)
        keypoints_raw:  numpy array of shape (17, 3) with [x, y, conf]
        conf_threshold: minimum confidence to draw a keypoint
    """
    if keypoints_raw is None or len(keypoints_raw) < 5:
        return

    kp = keypoints_raw
    has_conf = kp.shape[1] >= 3

    # Collect valid points
    points = {}
    for idx in range(min(17, len(kp))):
        x, y = int(kp[idx][0]), int(kp[idx][1])
        conf = float(kp[idx][2]) if has_conf else 1.0
        if conf >= conf_threshold and x > 0 and y > 0:
            points[idx] = (x, y, conf)

    # Draw bones first (under the keypoint circles)
    for (i, j) in _SKELETON_BONES:
        if i in points and j in points:
            region = _bone_region(i, j)
            color = _BONE_COLORS.get(region, (200, 200, 200))
            pt1 = (points[i][0], points[i][1])
            pt2 = (points[j][0], points[j][1])
            cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

    # Draw keypoint circles on top
    for idx, (x, y, conf) in points.items():
        # Color: bright green if high confidence, yellow if lower
        if conf >= 0.6:
            kp_color = (0, 255, 0)    # green
        elif conf >= 0.4:
            kp_color = (0, 255, 255)  # yellow
        else:
            kp_color = (0, 140, 255)  # orange
        radius = 4 if idx <= 4 else 5  # smaller dots for face keypoints
        cv2.circle(frame, (x, y), radius, kp_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), radius, (0, 0, 0), 1, cv2.LINE_AA)  # black outline


# ============================================================
#  Processing Thread — runs detection/tracking/scoring
# ============================================================

class ProcessingThread(threading.Thread):
    """
    Background thread that processes video frames through the
    full pipeline: YOLO detection → tracking → head pose →
    scoring → reliability checks → alert generation.

    Communicates with the main thread via two queues:
      frame_queue  — sends annotated frames for display
      alert_queue  — sends new alert records for the table + popup
    """

    def __init__(self, source, frame_queue, alert_queue, stop_event):
        super().__init__(daemon=True)
        self.source = source           # file path (str) or camera index (int)
        self.frame_queue = frame_queue
        self.alert_queue = alert_queue
        self.stop_event = stop_event   # threading.Event — set to stop processing

        # ---- Initialize all processing components ----
        # Dual-model approach:
        #   pose_model  → person detection + body keypoints + tracking
        #   phone_model → phone detection only (class 67)
        self.pose_model = load_pose_model()
        self.phone_model = load_model()
        self.scoring = ScoringEngine()
        self.persistence = FramePersistence()
        self.cooldown = AlertCooldown()

        # Store previous keypoints per track_id for hand velocity calculation
        self.prev_keypoints = {}

        # ---- Performance: frame skipping for phone model ----
        self.frame_count = 0
        self.cached_phones = []  # reuse last phone detection result between skips

        # ---- Yaw baseline calibration ----
        # For each student, we collect yaw readings for the first N frames
        # to establish what "forward" means from this camera angle.
        # After calibration, we measure DEVIATION from the baseline
        # instead of using the absolute yaw value.
        # This fixes the tilted-camera false-positive problem.
        self.yaw_baselines = {}    # tid → float (calibrated baseline yaw)
        self.yaw_samples = {}      # tid → list of yaw readings during calibration

    def run(self):
        """Main processing loop — runs until stop_event is set or video ends."""
        try:
            cap = cv2.VideoCapture(self.source)

            if not cap.isOpened():
                self.alert_queue.put({
                    "type": "error",
                    "message": f"Could not open video source: {self.source}"
                })
                return

            read_count = 0  # counts every frame read from the video

            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    # Video ended (for files) or camera error
                    self.frame_queue.put(None)  # signal "done" to the UI
                    break

                # ---- Frame skipping: only process every Nth frame ----
                read_count += 1
                if read_count % PROCESS_EVERY_N_FRAMES != 0:
                    continue  # skip this frame entirely

                frame_start = time.time()

                # ---- Step 1: YOLO Pose detection + tracking (persons + keypoints) ----
                persons = detect_poses(self.pose_model, frame)

                # ---- Step 2: YOLO Detection for phones (with frame skipping) ----
                if self.frame_count % PHONE_SKIP_FRAMES == 0:
                    self.cached_phones = detect_phones(self.phone_model, frame)
                phones = self.cached_phones
                self.frame_count += 1

                # ---- Step 3: Filter background people ----
                persons = filter_background_persons(persons)

                # ---- Step 4: Phone association ----
                phone_holders = associate_phones_to_students(persons, phones)

                # Active track IDs (for cleanup later)
                active_ids = {p["track_id"] for p in persons}

                # Build neighbor boxes list (needed for leaning/reaching detection)
                all_boxes = {p["track_id"]: p["box"] for p in persons}

                # ---- Step 5: Process each tracked student ----
                for person in persons:
                    tid = person["track_id"]
                    box = person["box"]
                    x1, y1, x2, y2 = box

                    # --- Head pose (from YOLO Pose face keypoints) ---
                    kp_raw = person.get("keypoints")
                    head = get_head_pose(keypoints=kp_raw)

                    # --- Yaw baseline calibration ---
                    if head["face_detected"]:
                        raw_yaw = head["yaw"]

                        if tid not in self.yaw_baselines:
                            if tid not in self.yaw_samples:
                                self.yaw_samples[tid] = []
                            self.yaw_samples[tid].append(raw_yaw)

                            if len(self.yaw_samples[tid]) >= YAW_BASELINE_FRAMES:
                                self.yaw_baselines[tid] = sorted(self.yaw_samples[tid])[
                                    len(self.yaw_samples[tid]) // 2
                                ]
                                del self.yaw_samples[tid]

                            yaw_deviation = 0.0
                        else:
                            yaw_deviation = abs(raw_yaw - self.yaw_baselines[tid])
                    else:
                        yaw_deviation = 0.0

                    # --- Frame persistence filtering (all signals) ---
                    raw_phone = tid in phone_holders
                    raw_look_away = (head["face_detected"] and
                                     yaw_deviation > YAW_THRESHOLD_DEGREES)
                    raw_face_missing = not head["face_detected"]

                    # --- Body pose analysis (from YOLO Pose keypoints) ---
                    parsed_kp = parse_keypoints(kp_raw)

                    neighbor_boxes = [b for t, b in all_boxes.items() if t != tid]

                    raw_leaning = detect_leaning(parsed_kp, box, neighbor_boxes)
                    raw_reaching = detect_reaching(parsed_kp, box, neighbor_boxes)

                    prev_kp = self.prev_keypoints.get(tid)
                    raw_hand_moving = detect_hand_movement(parsed_kp, prev_kp)
                    self.prev_keypoints[tid] = parsed_kp

                    raw_posture_off = detect_posture_deviation(parsed_kp)

                    confirmed_phone = self.persistence.update(tid, "phone", raw_phone)
                    confirmed_look_away = self.persistence.update(tid, "look_away", raw_look_away)
                    confirmed_face_missing = self.persistence.update(tid, "face_missing", raw_face_missing)
                    confirmed_leaning = self.persistence.update(tid, "leaning", raw_leaning)
                    confirmed_reaching = self.persistence.update(tid, "reaching", raw_reaching)
                    confirmed_hand_moving = self.persistence.update(tid, "hand_movement", raw_hand_moving)
                    confirmed_posture_off = self.persistence.update(tid, "posture_deviation", raw_posture_off)

                    effective_head = dict(head)
                    if not confirmed_look_away and head["face_detected"]:
                        effective_head["yaw"] = 0.0
                    if not confirmed_face_missing:
                        effective_head["face_detected"] = True

                    # --- Scoring (all 7 rules) ---
                    result = self.scoring.update(
                        tid,
                        has_phone=confirmed_phone,
                        head_pose=effective_head,
                        is_leaning=confirmed_leaning,
                        is_reaching=confirmed_reaching,
                        is_hand_moving=confirmed_hand_moving,
                        is_posture_off=confirmed_posture_off,
                    )

                    # --- Alert handling ---
                    if result["alert"] and self.cooldown.can_alert(tid):
                        self.cooldown.record_alert(tid)

                        screenshot_path = capture_screenshot(frame, tid, box)

                        alert_record = {
                            "track_id": tid,
                            "behavior": ", ".join(result["behaviors"]),
                            "confidence": round(result["score"], 1),
                            "screenshot_path": screenshot_path,
                            "timestamp": datetime.now().isoformat(),
                        }

                        add_alert(alert_record)
                        self.alert_queue.put({"type": "alert", "data": alert_record})
                    elif result["alert"]:
                        self.scoring.reset_alert_flag(tid)

                    # ---- Draw bounding box on the frame ----
                    score = result["score"]
                    behaviors = result["behaviors"]

                    if score >= CHEATING_THRESHOLD:
                        color = (0, 0, 255)
                    elif behaviors:
                        color = (0, 200, 255)
                    else:
                        color = (0, 255, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    label_parts = [f"ID:{tid}", f"Score:{score:.0f}"]
                    if behaviors:
                        label_parts.append(" | ".join(behaviors))
                    label = "  ".join(label_parts)

                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 2, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    draw_skeleton(frame, kp_raw)

                # ---- Draw phone boxes ----
                for phone in phones:
                    px1, py1, px2, py2 = phone["box"]
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 255), 2)
                    cv2.putText(frame, "PHONE", (px1, py1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                # ---- Cleanup stale tracking data ----
                self.scoring.cleanup_stale(active_ids)
                self.persistence.cleanup(active_ids)
                self.cooldown.cleanup(active_ids)

                if self.frame_queue.qsize() < 3:
                    self.frame_queue.put(frame)

            cap.release()

        except Exception as e:
            self.alert_queue.put({
                "type": "error",
                "message": f"Processing Error: {str(e)}"
            })
            self.frame_queue.put(None)
# ============================================================
#  Toast Notification — auto-dismissing alert (no OK button)
# ============================================================

class ToastNotification:
    """
    A sleek, auto-dismissing notification that slides in from the
    top-right corner.  No clicking required — it disappears after
    a few seconds.  Multiple toasts stack vertically.
    """
    _active_toasts = []  # class-level list to stack toasts

    def __init__(self, parent, title, message, duration_ms=4000,
                 bg_color="#ff6b6b", accent_color="#ee5a24"):
        self.parent = parent
        self.duration = duration_ms

        # Calculate vertical offset based on existing toasts
        offset_y = 10 + len(ToastNotification._active_toasts) * 95
        ToastNotification._active_toasts.append(self)

        # Create the toast frame
        self.frame = tk.Frame(
            parent, bg=bg_color, highlightbackground=accent_color,
            highlightthickness=2, cursor="hand2",
        )
        self.frame.place(relx=1.0, x=-10, y=offset_y, anchor="ne", width=360)

        # Left accent stripe
        stripe = tk.Frame(self.frame, bg=accent_color, width=5)
        stripe.pack(side="left", fill="y")

        # Content area
        content = tk.Frame(self.frame, bg=bg_color, padx=12, pady=10)
        content.pack(side="left", fill="both", expand=True)

        # Title
        tk.Label(
            content, text=title,
            font=("Segoe UI", 10, "bold"), bg=bg_color, fg="white",
            anchor="w",
        ).pack(fill="x")

        # Message
        tk.Label(
            content, text=message,
            font=("Segoe UI", 9), bg=bg_color, fg="#ffe0e0",
            anchor="w", wraplength=310, justify="left",
        ).pack(fill="x", pady=(2, 0))

        # Click to dismiss
        self.frame.bind("<Button-1>", lambda e: self._dismiss())
        for child in content.winfo_children():
            child.bind("<Button-1>", lambda e: self._dismiss())

        # Auto-dismiss after duration
        self._after_id = parent.after(duration_ms, self._dismiss)

    def _dismiss(self):
        """Remove this toast and restack remaining ones."""
        try:
            self.parent.after_cancel(self._after_id)
        except (ValueError, AttributeError):
            pass
        try:
            self.frame.destroy()
        except tk.TclError:
            pass
        if self in ToastNotification._active_toasts:
            ToastNotification._active_toasts.remove(self)
            # Restack remaining toasts
            for i, toast in enumerate(ToastNotification._active_toasts):
                try:
                    toast.frame.place_configure(y=10 + i * 95)
                except tk.TclError:
                    pass


# ============================================================
#  Main Tkinter Application — Premium Dark Theme
# ============================================================

# ---- Color Palette ----
COLORS = {
    "bg_primary": "#0c0a0a",      # deepest background (slightly warm dark)
    "bg_secondary": "#151212",    # panels / cards
    "bg_tertiary": "#1e1a1a",     # elevated surfaces
    "bg_header": "#100d0d",       # header bar
    "accent_blue": "#fca311",     # primary accent (Cheetah Gold)
    "accent_green": "#34d399",    # success / camera
    "accent_red": "#ef233c",      # danger / stop
    "accent_amber": "#fca311",    # warning / alerts (Gold)
    "accent_purple": "#e5e5e5",   # screenshot button (neutral light)
    "text_primary": "#ffffff",    # main text
    "text_secondary": "#a09c9c",  # muted text
    "text_dim": "#666060",        # very muted
    "border": "#2b2525",          # subtle borders
    "row_alt": "#1a1616",         # alternating row
    "selected": "#3b2a0c",        # selected row (dark gold)
    "hover_blue": "#ffb732",      # hover gold
    "hover_green": "#4ee6ad",
    "hover_red": "#ff4d63",
}


class ExamCheatingApp:
    """
    The main application window — premium dark theme with
    auto-dismissing toast notifications and polished visuals.
    """

    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1100x860")
        self.root.configure(bg=COLORS["bg_primary"])
        self.root.minsize(900, 650)

        # ---- State ----
        self.frame_queue = queue.Queue(maxsize=5)
        self.alert_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.processing_thread = None
        self.current_photo = None
        self.alert_count = 0  # running alert counter

        # ---- Build UI ----
        self._setup_styles()
        self._build_header()
        self._build_video_display()
        self._build_alerts_panel()
        self._build_status_bar()

        # ---- Start polling the queues ----
        self._poll_queues()

    # ---- Styling ----

    def _setup_styles(self):
        """Configure ttk styles for the entire app."""
        style = ttk.Style()
        style.theme_use("clam")

        # Treeview
        style.configure("Alerts.Treeview",
                        background=COLORS["bg_secondary"],
                        foreground=COLORS["text_primary"],
                        fieldbackground=COLORS["bg_secondary"],
                        font=("Segoe UI", 9),
                        rowheight=30,
                        borderwidth=0)
        style.configure("Alerts.Treeview.Heading",
                        background=COLORS["bg_tertiary"],
                        foreground=COLORS["accent_amber"],
                        font=("Segoe UI", 9, "bold"),
                        borderwidth=0,
                        relief="flat")
        style.map("Alerts.Treeview",
                  background=[("selected", COLORS["selected"])],
                  foreground=[("selected", "#ffffff")])

        # Scrollbar
        style.configure("Vertical.TScrollbar",
                        background=COLORS["bg_tertiary"],
                        troughcolor=COLORS["bg_secondary"],
                        borderwidth=0,
                        arrowsize=14)

    def _make_button(self, parent, text, bg, hover_bg, command, **kwargs):
        """Create a styled button with hover effects."""
        btn = tk.Button(
            parent, text=text, bg=bg, fg="white",
            font=("Segoe UI", 10, "bold"),
            relief="flat", cursor="hand2",
            padx=18, pady=7,
            activebackground=hover_bg, activeforeground="white",
            bd=0, highlightthickness=0,
            command=command, **kwargs,
        )
        # Hover animation
        btn.bind("<Enter>", lambda e: btn.config(bg=hover_bg))
        btn.bind("<Leave>", lambda e: btn.config(bg=bg))
        return btn

    # ---- UI Construction ----

    def _build_header(self):
        """Premium header bar with title + control buttons."""
        header = tk.Frame(self.root, bg=COLORS["bg_header"], pady=12, padx=16)
        header.pack(fill="x", side="top")

        # Accent line at the top
        accent_line = tk.Frame(self.root, bg=COLORS["accent_blue"], height=2)
        accent_line.pack(fill="x", side="top")

        # Left side: Title/Logo
        title_frame = tk.Frame(header, bg=COLORS["bg_header"])
        title_frame.pack(side="left")

        # Load logo image
        try:
            from config import resource_path
            logo_img = Image.open(resource_path("assets/logo.png"))
            logo_img.thumbnail((45, 45))
            self.logo_photo = ImageTk.PhotoImage(logo_img)
            logo_label = tk.Label(title_frame, image=self.logo_photo, bg=COLORS["bg_header"])
            logo_label.pack(side="left", padx=(0, 10))
        except Exception:
            # Fallback if logo not found
            tk.Label(
                title_frame, text="🐆",
                font=("Segoe UI", 24), bg=COLORS["bg_header"], fg=COLORS["accent_blue"],
            ).pack(side="left", padx=(0, 8))

        title_text = tk.Frame(title_frame, bg=COLORS["bg_header"])
        title_text.pack(side="left")

        tk.Label(
            title_text, text="Cheet Catcher",
            font=("Segoe UI", 16, "bold", "italic"), bg=COLORS["bg_header"],
            fg=COLORS["text_primary"],
        ).pack(anchor="w")

        tk.Label(
            title_text, text="AI-Powered Exam Monitoring",
            font=("Segoe UI", 8), bg=COLORS["bg_header"],
            fg=COLORS["accent_blue"],
        ).pack(anchor="w")

        # Right side: Buttons
        btn_frame = tk.Frame(header, bg=COLORS["bg_header"])
        btn_frame.pack(side="right")

        self.btn_stop = self._make_button(
            btn_frame, "⏹  Stop", COLORS["accent_red"], COLORS["hover_red"],
            self._on_stop, state="disabled",
        )
        self.btn_stop.pack(side="right", padx=(8, 0))

        self.btn_camera = self._make_button(
            btn_frame, "📷  Start Camera", COLORS["accent_green"],
            COLORS["hover_green"], self._on_camera,
        )
        self.btn_camera.pack(side="right", padx=(8, 0))

        self.btn_upload = self._make_button(
            btn_frame, "📂  Upload Video", COLORS["accent_blue"],
            COLORS["hover_blue"], self._on_upload,
        )
        self.btn_upload.pack(side="right")

        # Alert counter badge
        self.alert_badge = tk.Label(
            btn_frame, text="0 alerts",
            font=("Segoe UI", 9), bg=COLORS["bg_header"],
            fg=COLORS["text_dim"],
        )
        self.alert_badge.pack(side="right", padx=(0, 16))

    def _build_video_display(self):
        """Center area where annotated video frames are shown."""
        # Outer container with subtle border
        container = tk.Frame(
            self.root, bg=COLORS["border"], padx=1, pady=1,
        )
        container.pack(fill="both", expand=True, padx=12, pady=(10, 5))

        self.video_frame = tk.Frame(container, bg=COLORS["bg_primary"])
        self.video_frame.pack(fill="both", expand=True)

        self.video_label = tk.Label(
            self.video_frame, bg=COLORS["bg_primary"],
            text="\n\n📹  No video source selected\n\n"
                 "Click  Upload Video  or  Start Camera  to begin monitoring",
            font=("Segoe UI", 13), fg=COLORS["text_dim"],
            compound="top",
        )
        self.video_label.pack(fill="both", expand=True)

    def _build_alerts_panel(self):
        """Bottom panel with the alerts Treeview table + action buttons."""
        # Container with border
        container = tk.Frame(
            self.root, bg=COLORS["border"], padx=1, pady=1,
        )
        container.pack(fill="x", padx=12, pady=(5, 8))

        panel = tk.Frame(container, bg=COLORS["bg_secondary"], padx=10, pady=8)
        panel.pack(fill="x")

        # Header row
        header_frame = tk.Frame(panel, bg=COLORS["bg_secondary"])
        header_frame.pack(fill="x", pady=(0, 6))

        tk.Label(
            header_frame, text="⚠  Alerts Log",
            font=("Segoe UI", 11, "bold"), bg=COLORS["bg_secondary"],
            fg=COLORS["accent_amber"],
        ).pack(side="left")

        # Action buttons
        self.btn_screenshot = self._make_button(
            header_frame, "🖼  View Screenshot",
            COLORS["accent_purple"], "#b99cff",
            self._on_view_screenshot,
        )
        self.btn_screenshot.config(font=("Segoe UI", 9, "bold"), padx=12, pady=4)
        self.btn_screenshot.pack(side="right")

        self.btn_clear = self._make_button(
            header_frame, "🗑  Clear",
            COLORS["bg_tertiary"], "#2a2a50",
            self._on_clear_alerts,
        )
        self.btn_clear.config(font=("Segoe UI", 9, "bold"), padx=12, pady=4,
                              fg=COLORS["text_secondary"])
        self.btn_clear.pack(side="right", padx=(0, 6))

        # Treeview table
        cols = ("timestamp", "student_id", "confidence", "screenshot")

        tree_frame = tk.Frame(panel, bg=COLORS["bg_secondary"])
        tree_frame.pack(fill="x")

        self.alerts_tree = ttk.Treeview(
            tree_frame, columns=cols, show="headings",
            height=5, style="Alerts.Treeview",
        )

        self.alerts_tree.heading("timestamp",   text="⏰ Timestamp")
        self.alerts_tree.heading("student_id",  text="👤 Student")
        self.alerts_tree.heading("confidence",  text="📊 Score")
        self.alerts_tree.heading("screenshot",  text="📸 Screenshot")

        self.alerts_tree.column("timestamp",   width=170, anchor="center")
        self.alerts_tree.column("student_id",  width=100, anchor="center")
        self.alerts_tree.column("confidence",  width=100, anchor="center")
        self.alerts_tree.column("screenshot",  width=300, anchor="w")

        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical",
                                  command=self.alerts_tree.yview)
        self.alerts_tree.configure(yscrollcommand=scrollbar.set)

        self.alerts_tree.pack(side="left", fill="x", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Double-click a row to view the screenshot
        self.alerts_tree.bind("<Double-1>", lambda e: self._on_view_screenshot())

        # Alternating row colors via tags
        self.alerts_tree.tag_configure("odd", background=COLORS["bg_secondary"])
        self.alerts_tree.tag_configure("even", background=COLORS["row_alt"])

    def _build_status_bar(self):
        """Bottom status bar showing connection info."""
        status_frame = tk.Frame(self.root, bg=COLORS["bg_primary"], padx=12, pady=4)
        status_frame.pack(fill="x", side="bottom")

        self.status_dot = tk.Label(
            status_frame, text="●",
            font=("Segoe UI", 8), bg=COLORS["bg_primary"], fg=COLORS["text_dim"],
        )
        self.status_dot.pack(side="left")

        self.status_var = tk.StringVar(value="Ready  |  Checking Firebase...")
        tk.Label(
            status_frame, textvariable=self.status_var,
            font=("Segoe UI", 8), bg=COLORS["bg_primary"],
            fg=COLORS["text_dim"], anchor="w",
        ).pack(side="left", padx=(4, 0))

        # Version label on the right
        tk.Label(
            status_frame, text="v2.0",
            font=("Segoe UI", 8), bg=COLORS["bg_primary"],
            fg=COLORS["text_dim"],
        ).pack(side="right")

        # Update Firebase status after a short delay
        self.root.after(500, self._update_firebase_status)

    def _update_firebase_status(self):
        """Check whether Firebase connected and update the status bar."""
        from firebase_db import _firestore_db
        if _firestore_db is not None:
            self.status_var.set("Ready  |  Firebase: Connected")
            self.status_dot.config(fg=COLORS["accent_green"])
        else:
            self.status_var.set("Ready  |  Local Storage Mode")
            self.status_dot.config(fg=COLORS["accent_amber"])

    # ---- Event Handlers ----

    def _on_upload(self):
        """Handle 'Upload Video' button click."""
        filepath = filedialog.askopenfilename(
            title="Select Exam Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv"),
                ("All files", "*.*"),
            ],
        )
        if filepath:
            self._start_processing(filepath)

    def _on_camera(self):
        """Handle 'Start Camera' button click."""
        self._start_processing(0)

    def _on_stop(self):
        """Handle 'Stop' button click — cleanly stop processing."""
        self.stop_event.set()
        self.btn_stop.config(state="disabled", bg=COLORS["bg_tertiary"])
        self.btn_upload.config(state="normal")
        self.btn_camera.config(state="normal")
        self.status_var.set("Stopped  |  Ready for new source")
        self.status_dot.config(fg=COLORS["text_dim"])

    def _on_view_screenshot(self):
        """Open the screenshot for the selected alert row in a new window."""
        selected = self.alerts_tree.selection()
        if not selected:
            ToastNotification(
                self.root, "ℹ️ No Selection",
                "Please select an alert row first.",
                bg_color="#3b3b5c", accent_color=COLORS["accent_blue"],
                duration_ms=2500,
            )
            return

        values = self.alerts_tree.item(selected[0], "values")
        screenshot_path = values[3]  # timestamp(0), student(1), score(2), screenshot(3)

        if not os.path.exists(screenshot_path):
            ToastNotification(
                self.root, "❌ File Not Found",
                f"Screenshot not found: {screenshot_path}",
                bg_color="#5c2020", accent_color=COLORS["accent_red"],
                duration_ms=3000,
            )
            return

        # Open in a styled Toplevel window
        viewer = tk.Toplevel(self.root)
        viewer.title(f"Screenshot — Student {values[1]}")
        viewer.configure(bg=COLORS["bg_primary"])

        img = Image.open(screenshot_path)
        img.thumbnail((900, 600))
        photo = ImageTk.PhotoImage(img)

        # Image with border
        img_container = tk.Frame(viewer, bg=COLORS["border"], padx=1, pady=1)
        img_container.pack(padx=16, pady=(16, 8))

        label = tk.Label(img_container, image=photo, bg=COLORS["bg_primary"])
        label.image = photo
        label.pack()

        info = tk.Label(
            viewer,
            text=f"Student {values[1]}  •  Score: {values[2]}  •  {values[0]}",
            font=("Segoe UI", 10), bg=COLORS["bg_primary"],
            fg=COLORS["text_secondary"],
        )
        info.pack(pady=(0, 16))

    def _on_clear_alerts(self):
        """Clear all rows from the alerts table."""
        for item in self.alerts_tree.get_children():
            self.alerts_tree.delete(item)
        self.alert_count = 0
        self.alert_badge.config(text="0 alerts", fg=COLORS["text_dim"])

    # ---- Processing Control ----

    def _start_processing(self, source):
        """
        Start the background processing thread with the given source.
        """
        if self.processing_thread and self.processing_thread.is_alive():
            self.stop_event.set()
            self.processing_thread.join(timeout=2)

        self.stop_event.clear()
        self.frame_queue = queue.Queue(maxsize=5)
        self.alert_queue = queue.Queue()

        self.btn_upload.config(state="disabled")
        self.btn_camera.config(state="disabled")
        self.btn_stop.config(state="normal", bg=COLORS["accent_red"])

        source_name = os.path.basename(source) if isinstance(source, str) else "Live Camera"
        self.status_var.set(f"▶ Processing: {source_name}")
        self.status_dot.config(fg=COLORS["accent_green"])

        self.processing_thread = ProcessingThread(
            source=source,
            frame_queue=self.frame_queue,
            alert_queue=self.alert_queue,
            stop_event=self.stop_event,
        )
        self.processing_thread.start()

    # ---- Queue Polling (main thread only) ----

    def _poll_queues(self):
        """
        Called every QUEUE_POLL_MS by root.after().
        Checks both queues and updates the UI accordingly.
        """
        # ---- Check frame queue ----
        try:
            frame = self.frame_queue.get_nowait()
            if frame is None:
                self._on_stop()
                self.video_label.config(
                    image="",
                    text="\n\n✅  Video processing complete",
                    font=("Segoe UI", 14), fg=COLORS["accent_green"],
                )
            else:
                self._display_frame(frame)
        except queue.Empty:
            pass

        # ---- Check alert queue ----
        try:
            while True:
                msg = self.alert_queue.get_nowait()
                if msg["type"] == "alert":
                    self._handle_new_alert(msg["data"])
                elif msg["type"] == "error":
                    ToastNotification(
                        self.root, "❌ Error", msg["message"],
                        bg_color="#5c2020", accent_color=COLORS["accent_red"],
                    )
        except queue.Empty:
            pass

        self.root.after(QUEUE_POLL_MS, self._poll_queues)

    def _display_frame(self, frame):
        """
        Convert an OpenCV BGR frame to a Tkinter-compatible image
        and display it in the video label.
        """
        h, w = frame.shape[:2]
        scale = min(VIDEO_MAX_WIDTH / w, VIDEO_MAX_HEIGHT / h, 1.0)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(img)

        self.video_label.config(image=photo, text="")
        self.current_photo = photo

    def _handle_new_alert(self, alert_data):
        """
        Insert a new alert row into the Treeview and show a toast notification.
        Called from the main thread only (via queue polling).
        """
        try:
            dt = datetime.fromisoformat(alert_data["timestamp"])
            display_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, KeyError):
            display_time = alert_data.get("timestamp", "N/A")

        # Alternating row tags
        self.alert_count += 1
        tag = "odd" if self.alert_count % 2 else "even"

        self.alerts_tree.insert("", 0, values=(
            display_time,
            f"#{alert_data['track_id']}",
            alert_data["confidence"],
            alert_data.get("screenshot_path", "N/A"),
        ), tags=(tag,))

        # Update the alert counter badge
        self.alert_badge.config(
            text=f"{self.alert_count} alert{'s' if self.alert_count != 1 else ''}",
            fg=COLORS["accent_red"],
        )

        # Show a toast notification (auto-dismisses, no click needed)
        ToastNotification(
            self.root,
            f"⚠️  Student #{alert_data['track_id']} — Cheating Alert",
            f"{alert_data['behavior']}  •  Score: {alert_data['confidence']}",
            duration_ms=4000,
            bg_color="#4a1a1a",
            accent_color=COLORS["accent_red"],
        )


# ============================================================
#  Application Entry Point
# ============================================================

def main():
    root = tk.Tk()
    app = ExamCheatingApp(root)

    # Handle window close
    def on_closing():
        app.stop_event.set()
        if app.processing_thread and app.processing_thread.is_alive():
            app.processing_thread.join(timeout=2)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
