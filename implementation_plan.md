# Exam Cheating Detection — Implementation Plan

> **Based on:** [Exam_Cheating_Detection_Master_Plan.md](file:///A:/projects/Exam Cheating Detection/Exam_Cheating_Detection_Master_Plan.md)

This plan covers building a desktop AI application that watches exam video (uploaded or live camera), tracks students individually, flags suspicious behavior (phone use, looking away, face disappearing), draws colored bounding boxes, captures incident screenshots, logs to Firebase (with local fallback), and shows real-time alerts in a Tkinter GUI.

## User Review Required

> [!IMPORTANT]
> **This is a very large project (20 phases).** I can implement Phases 0–15 (the full working application) in code. However, some phases require **your manual action**:
> - **Phase 10 (Firebase Setup):** You need to create a Firebase project in the console and provide `firebase_credentials.json`. I'll write code that gracefully falls back to local JSON if credentials aren't available.
> - **Phase 16 (Testing & Threshold Tuning):** Requires running against a real exam video and manually tuning values — I'll set sensible defaults.
> - **Phase 17 (Unit Tests):** Marked optional in the plan. I'll include them.
> - **Phase 18 (Portfolio Packaging):** README, demo media, LinkedIn post — I'll create the README.
> - **Phase 19 (Standalone .exe):** Requires PyInstaller build — I'll set up the config but you'll run the build locally.

> [!WARNING]
> **No sample exam video is included.** The app will work with any video file or your webcam. For testing, you'll need to provide or record a short exam-like video.

## Open Questions

> [!IMPORTANT]
> 1. **Python version:** Do you have Python 3.10+ installed? The project needs it for `ultralytics` and `mediapipe`.
> 2. **YOLO model:** The plan says `yolov8n.pt`. Should I use YOLOv8n (stable) or YOLO11n (newer)? I'll default to **YOLOv8n** as the plan specifies.
> 3. **Firebase:** Do you want me to skip Firebase integration for now (use local JSON only) and you'll configure Firebase later? Or do you already have a Firebase project ready?

---

## Proposed Changes

### Phase 0 & 1: Project Structure & Setup

#### [NEW] [config.py](file:///A:/projects/Exam Cheating Detection/config.py)
Central configuration file with ALL tunable thresholds and paths. Includes model path, confidence thresholds, scoring parameters, cooldown settings, head pose thresholds, and directory paths.

#### [NEW] [requirements.txt](file:///A:/projects/Exam Cheating Detection/requirements.txt)
Dependencies: `ultralytics`, `opencv-python`, `mediapipe`, `pillow`, `firebase-admin`, `numpy`.

#### [NEW] Folder structure
Create `models/`, `alerts/screenshots/`, `videos/`, `tests/`, `assets/` directories with `.gitkeep` placeholders.

#### [NEW] [.gitignore](file:///A:/projects/Exam Cheating Detection/.gitignore)
Ignore `firebase_credentials.json`, `models/*.pt`, `alerts/screenshots/`, `dist/`, `build/`, `__pycache__/`, `.venv/`.

---

### Phase 2 & 3: Detection & Tracking

#### [NEW] [detector.py](file:///A:/projects/Exam Cheating Detection/detector.py)
- Loads YOLOv8n pretrained model (auto-downloads on first run)
- Detects persons (class 0) and cell phones (class 67)
- Uses BoT-SORT tracking (`model.track(persist=True, tracker="botsort.yaml")`) for stable track IDs
- Returns structured detection results per frame with track IDs, bounding boxes, and class labels

---

### Phase 4: Phone Detection & Association

#### [MODIFY] [detector.py](file:///A:/projects/Exam Cheating Detection/detector.py)
- `expand_box()` function expands student box by 15% to check phone proximity
- `point_in_box()` function checks if phone center falls inside expanded student box
- `associate_phones_to_students()` links detected phones to their nearest tracked student

---

### Phase 5 & 6: Head Pose Detection

#### [NEW] [head_pose.py](file:///A:/projects/Exam Cheating Detection/head_pose.py)
- Uses MediaPipe Face Mesh to extract 3D facial landmarks
- Computes pitch/yaw/roll from nose tip, chin, eye corners using `cv2.solvePnP`
- `get_head_pose(crop)` function takes a cropped student region and returns angles
- Handles "no face found" gracefully (returns `face_missing = True`)

---

### Phase 7: Scoring Engine

#### [NEW] [scoring.py](file:///A:/projects/Exam Cheating Detection/scoring.py)
- Per-student state dictionary keyed by `track_id`: score, looking_away_since, face_missing_since
- Rules engine:
  - Phone detected → +100 points (instant)
  - Looking away (yaw > threshold) for > N seconds → +40 points
  - Face missing for > N seconds → +50 points
- Threshold crossing triggers alert **once per crossing** (not every frame)
- Score decay over time when behavior stops

---

### Phase 8: Reliability Layer

#### [NEW] [reliability.py](file:///A:/projects/Exam Cheating Detection/reliability.py)
- **Frame persistence:** Requires behavior to be true for `CONFIRM_FRAMES` consecutive frames before counting
- **Background filtering:** Ignores persons below `RELATIVE_MIN_HEIGHT_RATIO` of tallest person
- **Alert cooldown:** Prevents same student from triggering alerts within `ALERT_COOLDOWN_SECONDS`

---

### Phase 9: Screenshot Capture

#### [MODIFY] [scoring.py](file:///A:/projects/Exam Cheating Detection/scoring.py) / integrated into alert flow
- On alert trigger, captures full frame or cropped student region
- Saves to `alerts/screenshots/track{id}_{timestamp}.jpg`
- Returns file path for database logging

---

### Phase 11: Firebase + Local Fallback

#### [NEW] [firebase_db.py](file:///A:/projects/Exam Cheating Detection/firebase_db.py)
- `add_alert(record)` — tries Firebase first (upload screenshot to Storage, write Firestore doc), falls back to local JSON
- `get_all_alerts()` — reads from Firebase or local JSON, identical return format
- Local fallback: appends to `alerts/local_alerts.json` with same schema
- Never crashes regardless of Firebase availability

---

### Phase 12–15: Tkinter GUI with Threading

#### [NEW] [main.py](file:///A:/projects/Exam Cheating Detection/main.py)
- **Main window layout:**
  - Top: Video display canvas (Label with PhotoImage)
  - Control bar: "Upload Video" button, "Start Camera" button, "Stop" button
  - Bottom: Alerts table (Treeview) with columns: timestamp, student ID, behavior, confidence, screenshot path
  - "View Screenshot" button / double-click to open saved images
- **Threading architecture:**
  - Detection/tracking/scoring runs in a background `threading.Thread`
  - `queue.Queue` passes finished frames and alerts to main thread
  - `root.after(15)` polls the queue to update UI — never touches Tkinter from background thread
- **Alert popup:** `messagebox.showwarning()` on each new incident
- **Video sources:** File upload via `filedialog.askopenfilename()`, live camera via `cv2.VideoCapture(0)`

---

### Phase 17: Unit Tests

#### [NEW] [tests/test_scoring.py](file:///A:/projects/Exam Cheating Detection/tests/test_scoring.py)
- Tests for scoring rules (point additions, threshold crossing)
- Tests for bbox association (phone-to-student linking)
- Tests for reliability filters (frame persistence, cooldown)

---

### Phase 18: README

#### [NEW] [README.md](file:///A:/projects/Exam Cheating Detection/README.md)
- Problem statement, architecture overview, tech stack
- How the scoring engine works (the most original part)
- Reliability techniques from Phase 8
- Ethical framing: evidence for human review, no face recognition, anonymous track IDs
- Screenshots, setup instructions, download link for .exe

---

## Verification Plan

### Automated Tests
```bash
cd "A:\projects\Exam Cheating Detection"
python -m pytest tests/ -v
```

### Manual Verification
1. **Import test:** `python -c "import ultralytics, cv2, mediapipe"` — no errors
2. **Video processing:** Upload a sample video → green/red boxes appear, tracking IDs are stable
3. **Alert flow:** Simulate suspicious behavior → alert fires, screenshot saved, row appears in table
4. **Camera test:** Start webcam → same processing works live
5. **Fallback test:** Remove Firebase credentials → app still runs using local JSON
6. **UI responsiveness:** Window stays responsive (draggable, buttons clickable) during processing
