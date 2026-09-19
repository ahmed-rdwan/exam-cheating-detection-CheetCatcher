# Exam Cheating Detection System 🎓

A desktop AI application that monitors exam sessions via video (uploaded or live camera), tracks every student individually, and flags suspicious behavior in real-time — providing evidence for human review, not automated disciplinary decisions.

---

## 🎯 What It Does

| Feature | Description |
|---|---|
| **Person Detection & Tracking** | YOLOv8n detects every student; BoT-SORT assigns stable IDs across frames |
| **Phone Detection** | Detects cell phones and associates each to the nearest student |
| **Head Pose Estimation** | MediaPipe Face Mesh computes where each student is looking (yaw/pitch/roll) |
| **Scoring Engine** | Time-based rules turn frame-level signals into reliable decisions — one bad frame never triggers a false alert |
| **Reliability Layer** | Frame persistence, background filtering, and alert cooldown eliminate false positives |
| **Screenshot Evidence** | Auto-captures and saves a screenshot the moment an alert fires |
| **Real-Time Alerts** | In-app popup + alerts table with full incident log |
| **Firebase + Local Fallback** | Logs to Firestore when available; falls back to local JSON seamlessly |
| **Live Camera & File Upload** | Works with pre-recorded videos or a live webcam feed |

---

## 🧠 How the Scoring Engine Works

This is the most original part of the project — "cheating" is never a direct model output. Instead, the system uses a **point-based scoring engine** that accumulates evidence over time:

| Rule | Points | Condition |
|---|---|---|
| Phone detected near student | +100 | Phone bbox center inside expanded student bbox (confirmed by 3-frame persistence) |
| Looking away (yaw > 20°) | +40 | Sustained for ≥ 4 seconds |
| Face missing | +50 | Missing for ≥ 3 seconds |
| **Alert threshold** | **80** | When crossed, triggers screenshot + database log + UI alert |
| Score decay | -2/update | When no suspicious behavior is active |

**Key design decisions:**
- **Time-based, not frame-based:** A single bad frame (e.g., a pen misread as a phone for 1 frame) is filtered out by the 3-frame persistence requirement.
- **One alert per crossing:** The threshold triggers only once per crossing, not every frame the score stays high.
- **Cooldown:** After an alert, the same student can't trigger another for 10 seconds — prevents alert spam from a single ongoing incident.
- **Score decay:** Scores gradually return to zero when behavior stops, so past incidents don't permanently flag a student.

---

## 🛡️ Reliability Techniques

These three mechanisms are what make this project feel production-grade:

1. **Frame Persistence** — A detection must be consistent for 3 consecutive frames before it "counts," and must be absent for 3 frames before clearing. This alone removes most false positives.

2. **Background Filtering** — Ignores detected persons whose bounding-box height is below 60% of the tallest person in the frame. This filters out people walking in the background who aren't exam students.

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
| Detection | YOLOv8n (pretrained on COCO) |
| Tracking | BoT-SORT (via Ultralytics) |
| Head Pose | MediaPipe Face Mesh + cv2.solvePnP |
| GUI | Tkinter (threaded processing) |
| Database | Firebase Firestore (with local JSON fallback) |
| Language | Python 3.10+ |

---

## 🚀 Getting Started

### Option 1: Download and Run (no Python needed)
> ⬇️ **[Download ExamCheatingDetector.exe from Releases](../../releases/latest)**
>
> Just double-click to run — no installation required.

### Option 2: Run from Source

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/exam-cheating-detection.git
cd exam-cheating-detection

# Create a virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

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
exam_cheating_detection/
├── main.py                  # Tkinter GUI entry point
├── detector.py              # YOLO detection + tracking
├── head_pose.py             # MediaPipe head pose logic
├── scoring.py               # Scoring engine + screenshot capture
├── reliability.py           # Frame persistence, background filter, cooldown
├── firebase_db.py           # Firebase Firestore + local JSON fallback
├── config.py                # ALL tunable thresholds and paths
├── models/
│   └── yolov8n.pt           # auto-downloaded on first run
├── alerts/
│   ├── screenshots/         # saved incident images
│   └── local_alerts.json    # fallback when Firebase isn't configured
├── tests/
│   └── test_scoring.py      # unit tests for core logic
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
YAW_THRESHOLD_DEGREES = 20    # degrees of head turn to count as "looking away"
LOOK_AWAY_SECONDS = 4         # seconds before look-away adds points
CHEATING_THRESHOLD = 80       # score that triggers an alert
CONFIRM_FRAMES = 3            # frames of consistency required
ALERT_COOLDOWN_SECONDS = 10   # minimum gap between alerts for same student
```

---

## 📄 License

This project is for educational and portfolio purposes.
