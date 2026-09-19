# ---- Configuration file for Exam Cheating Detection ----
# All tunable thresholds and paths live here, so you never
# need to hunt through logic files to change a number.
# Organized by feature area for easy navigation.

import os
import sys

# ---- Helper: resource path (works both as .py and as PyInstaller .exe) ----
def resource_path(relative_path):
    """
    When running as a normal .py file, use the current folder.
    When running as a PyInstaller .exe, files are unpacked into a
    temporary folder referenced by sys._MEIPASS.
    """
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, relative_path)


# ---- YOLO Detection Settings ----
MODEL_PATH = resource_path("models/yolov8n.pt")       # detection model for phones (class 67)
POSE_MODEL_PATH = resource_path("models/yolo11n-pose.pt")  # pose model for persons + keypoints
CONF_THRESHOLD = 0.3       # lowered from 0.4 — catches more people in crowded scenes
PHONE_CONF_THRESHOLD = 0.15  # phones are small objects, need very low confidence to detect
PERSON_CLASS = 0           # COCO class ID for "person"
PHONE_CLASS = 67           # COCO class ID for "cell phone"

# ---- Tracking Settings ----
TRACKER_CONFIG = "botsort.yaml"  # alternatives: "bytetrack.yaml"

# ---- Phone-to-Student Association ----
PHONE_EXPAND_RATIO = 0.30  # how much to expand student box when checking phone proximity (wider catch area)

# ---- Head Pose Thresholds ----
# YAW_THRESHOLD is now measured as DEVIATION from each student's baseline,
# not as an absolute angle.  This fixes the tilted-camera problem:
# the first N frames establish what "forward" means for that student,
# then only changes from that baseline count as "looking away."
YAW_THRESHOLD_DEGREES = 22    # deviation from baseline to count as "looking away" (raised to reduce false positives)
PITCH_THRESHOLD_DEGREES = 25  # if |pitch| exceeds this, student is "looking down/up"
YAW_BASELINE_FRAMES = 30      # how many frames to use for calibrating each student's baseline yaw

# ---- Scoring Engine (original rules) ----
LOOK_AWAY_SECONDS = 5      # how long a student must look away before points are added
LOOK_AWAY_POINTS = 40      # points added when the look-away timer expires
FACE_MISSING_SECONDS = 4   # how long a face must be missing before points are added
FACE_MISSING_POINTS = 50   # points added when the face-missing timer expires
PHONE_POINTS = 100         # points added immediately when a phone is confirmed near a student
CHEATING_THRESHOLD = 80    # score at or above this triggers an alert
SCORE_DECAY_RATE = 5       # points subtracted per second when no suspicious behavior is active (faster forgiveness)

# ---- Body Pose Scoring (NEW — from YOLO Pose keypoints) ----
# Leaning: student's torso midpoint shifts toward a neighbor
LEAN_DISTANCE_RATIO = 0.4      # torso shift > 40% of own box width = "leaning" (less sensitive)
LEAN_SECONDS = 2               # must lean for 2 seconds before points are added
LEAN_POINTS = 35               # points added when the lean timer expires

# Reaching: student's wrist extends beyond their own bounding box toward another student
REACH_BEYOND_RATIO = 0.2       # wrist extends > 20% beyond own box edge
REACH_SECONDS = 1.5            # must reach for 1.5 seconds before points are added
REACH_POINTS = 45              # points added when the reach timer expires

# Excessive hand movement: wrist moves too fast / too much across frames
HAND_VELOCITY_THRESHOLD = 80   # pixels/frame wrist displacement to count as "excessive" (raised — normal writing ~40-60px)
HAND_MOVEMENT_SECONDS = 3      # sustained excessive movement before points are added
HAND_MOVEMENT_POINTS = 30      # points added when the hand-movement timer expires

# Posture deviation: shoulder-hip line tilts away from vertical (not writing posture)
POSTURE_ANGLE_THRESHOLD = 35   # degrees from vertical to count as "off posture" (raised — normal sitting varies)
POSTURE_SECONDS = 3            # must be off-posture for 3 seconds before points are added
POSTURE_POINTS = 30            # points added when the posture timer expires

# ---- Reliability / Professional Touches ----
CONFIRM_FRAMES = 3                  # behavior must persist this many consecutive frames to count
RELATIVE_MIN_HEIGHT_RATIO = 0.12    # very low — keeps back-row students visible in classroom settings
ALERT_COOLDOWN_SECONDS = 10         # minimum seconds between two alerts for the same student

# ---- Performance Settings ----
PHONE_SKIP_FRAMES = 3               # run phone detection every Nth frame (saves ~33% inference time)
POSE_IMGSZ = 480                    # input resolution for pose model (lower = faster, 640 default)
PHONE_IMGSZ = 640                   # input resolution for phone model (full res — phones are tiny objects)

# ---- File Paths ----
SCREENSHOT_DIR = "alerts/screenshots"
LOCAL_ALERTS_JSON = "alerts/local_alerts.json"
VIDEO_OUTPUT_DIR = "videos"

# ---- Firebase Settings ----
# Set this to the path of your Firebase service-account JSON key.
# If the file doesn't exist, the app gracefully falls back to local JSON.
FIREBASE_CREDENTIALS_PATH = "firebase_credentials.json"
FIREBASE_COLLECTION = "cheating_alerts"

# ---- UI Settings ----
QUEUE_POLL_MS = 15          # how often (ms) the Tkinter main loop checks the frame queue
VIDEO_MAX_WIDTH = 900       # max width for the video display in the GUI
VIDEO_MAX_HEIGHT = 600      # max height for the video display in the GUI
APP_TITLE = "Cheet Catcher"
