# 🎓 CheetCatcher — Exam Cheating Detection System

A desktop AI application that monitors exam sessions via video (uploaded or live camera), tracks every student individually, and flags suspicious behavior in real time — providing **evidence for human review**, not automated disciplinary decisions.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/YOLO-Ultralytics-00FFFF?logo=yolo&logoColor=black" alt="Ultralytics YOLO">
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/UI-Tkinter-informational" alt="Tkinter">
  <img src="https://img.shields.io/badge/Firebase-Firestore-FFCA28?logo=firebase&logoColor=black" alt="Firebase">
  <img src="https://img.shields.io/badge/Platform-Windows-0078D6?logo=windows&logoColor=white" alt="Windows">
  <img src="https://img.shields.io/badge/status-active-success" alt="Status: Active">
</p>

---

## 📹 Demo

<video src="https://raw.githubusercontent.com/ahmed-rdwan/exam-cheating-detection-CheetCatcher/main/assets/demo.mp4" controls width="700"></video>

---

## 🎯 What It Does

| Feature | Description |
|---|---|
| **Person Detection & Tracking** | YOLO Pose detects every student and extracts 17 body keypoints; BoT-SORT assigns stable IDs across frames |
| **Phone Detection** | A second lightweight YOLO pass detects cell phones and associates each to the nearest student |
| **Head Pose Estimation** | Estimated directly from face keypoints (nose, eyes, ears) — no separate face-recognition model |
| **Body Posture Analysis** | Flags leaning toward a neighbor, reaching beyond your own desk, excessive hand movement, and off-normal writing posture |
| **Scoring Engine** | Time-based rules turn frame-level signals into reliable decisions — one bad frame never triggers a false alert |
| **Reliability Layer** | Frame persistence, background filtering, and alert cooldown eliminate false positives |
| **Screenshot Evidence** | Auto-captures and saves a screenshot the moment an alert fires |
| **Real-Time Alerts** | In-app toast notification + alerts table with full incident log |
| **Firebase + Local Fallback** | Logs to Firestore when available; falls back to local JSON seamlessly |
| **Live Camera & File Upload** | Works with pre-recorded videos or a live webcam feed |

---

## 🧠 How the Scoring Engine Works

This is the most original part of the project — "cheating" is never a direct model output. Instead, the system uses a **point-based scoring engine** that accumulates evidence over time, per student:

| Rule | Points | Condition |
|---|---|---|
| Phone detected near student | +100 | Phone bbox center inside expanded student bbox (confirmed by frame persistence) |
| Looking away (yaw beyond threshold) | +40 | Sustained for ≥ 5 seconds |
| Face missing | +50 | Missing for ≥ 4 seconds |
| Leaning toward a neighbor | +35 | Sustained for ≥ 2 seconds |
| Reaching beyond own desk | +45 | Sustained for ≥ 1.5 seconds |
| Excessive hand movement | +30 | Sustained for ≥ 3 seconds |
| Off-normal posture | +30 | Sustained for ≥ 3 seconds |
| **Alert threshold** | **80** | When crossed, triggers screenshot + database log + UI alert |
| Score decay | -5/cycle | When no suspicious behavior is active |

**Key design decisions:**
- **Time-based, not frame-based:** A single bad frame (e.g., a pen misread as a phone for one frame) is filtered out by the frame-persistence requirement.
- **One alert per crossing:** The threshold triggers only once per crossing, not every frame the score stays high.
- **Cooldown:** After an alert, the same student can't trigger another for 10 seconds — prevents alert spam from a single ongoing incident.
- **Score decay:** Scores gradually return to zero when behavior stops, so past incidents don't permanently flag a student.

---

## 🛡️ Reliability Techniques

These three mechanisms are what make this project feel production-grade:

1. **Frame Persistence** — A detection must be consistent for several consecutive frames before it "counts," and must be absent for the same number of frames before clearing. This alone removes most false positives.
2. **Background Filtering** — Ignores detected persons whose bounding-box height is well below the tallest person in the frame. This filters out people walking in the background who aren't exam students.
3. **Alert Cooldown** — After a student triggers an alert, no further alerts fire for that student for 10 seconds. Prevents the same incident from generating multiple alerts.

---

## ⚖️ Ethical Framing

> This system produces **evidence for human review**, not an automatic disciplinary decision.
>
> - No face recognition — students are identified only by anonymous track IDs
> - No stored student names or personally identifiable information
> - Alerts are flagged for a human invigilator to review, not acted upon automatically
> - The system is a tool to assist, not replace, human judgment

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Detection & Pose | YOLO Pose + YOLO Detection (Ultralytics) |
| Tracking | BoT-SORT |
| Head Pose | Estimated from YOLO Pose face keypoints |
| Body Pose | Estimated from YOLO Pose body keypoints |
| GUI | Tkinter (threaded processing) |
| Database | Firebase Firestore (with local JSON fallback) |
| Packaging | PyInstaller |
| Language | Python 3.10+ |

---

## 🚀 Getting Started

### Option 1: Download and Run (no Python needed)
> ⬇️ **[Download CheetCatcher from Releases](https://github.com/ahmed-rdwan/exam-cheating-detection-CheetCatcher/releases/latest)**
>
> Unzip and double-click `CheetCatcher.exe` — no installation required.

### Option 2: Run from Source

```bash
# Clone the repository
git clone https://github.com/ahmed-rdwan/exam-cheating-detection-CheetCatcher.git
cd exam-cheating-detection-CheetCatcher

# Create a virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Firebase Setup (Optional)

1. Create a Firebase project at [console.firebase.google.com](https://console.firebase.google.com)
2. Enable **Firestore Database**
3. Go to Project Settings → Service Accounts → Generate new private key
4. Save as `firebase_credentials.json` in the project root
5. The app will automatically detect and use it

> **Without Firebase:** The app works perfectly with local JSON storage — no setup needed.

---

## 📁 Project Structure

```
exam-cheating-detection-CheetCatcher/
├── main.py                  # Tkinter GUI entry point
├── detector.py               # YOLO detection + tracking
├── head_pose.py               # Head pose from keypoints
├── body_pose.py                # Body posture analysis from keypoints
├── scoring.py                # Scoring engine + screenshot capture
├── reliability.py             # Frame persistence, background filter, cooldown
├── firebase_db.py              # Firebase Firestore + local JSON fallback
├── config.py                 # ALL tunable thresholds and paths
├── models/                   # YOLO weights (auto-downloaded on first run)
├── alerts/
│   ├── screenshots/          # saved incident images
│   └── local_alerts.json     # fallback when Firebase isn't configured
├── tests/
│   └── test_scoring.py       # unit tests for core logic
├── requirements.txt
└── README.md
```

---

## 🧪 Testing

Core scoring and detection-association logic is covered by unit tests:

```bash
python -m pytest tests/ -v
```

---

## 📋 Configuration

All thresholds are centralized in [`config.py`](config.py) — tune them without touching any logic file:

```python
YAW_THRESHOLD_DEGREES = 22    # degrees of head turn to count as "looking away"
LOOK_AWAY_SECONDS = 5         # seconds before look-away adds points
CHEATING_THRESHOLD = 80       # score that triggers an alert
CONFIRM_FRAMES = 3            # frames of consistency required
ALERT_COOLDOWN_SECONDS = 10   # minimum gap between alerts for same student
```

---

## 📄 License

This project is for educational and portfolio purposes.
