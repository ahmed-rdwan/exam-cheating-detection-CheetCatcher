# Exam Cheating Tracking & Alert System — Project Master Plan

**Purpose of this document:** this is the single source of truth for the project. Each phase below is self-contained (goal, steps, code hints, deliverable), so you can paste **one phase at a time** into a new conversation and get full context without re-explaining the project.

**Goal of the project:** a desktop AI application that watches exam video (uploaded or live camera), tracks every student individually, flags suspicious behavior (phone use, looking away, face disappearing), draws a red box on flagged students / green on normal ones, captures a screenshot of every incident, logs it to Firebase, and shows a real-time alert in a Tkinter GUI.

**Confirmed decisions for this project:**
- Detection: YOLOv8n / YOLO11n, **pretrained on COCO** — no custom training, no dataset needed. Only `person` (class 0) and `cell phone` (class 67) are used.
- Tracking: BoT-SORT or ByteTrack (built into Ultralytics) — gives each student a stable `track_id`.
- Head pose: MediaPipe Face Mesh (pitch/yaw/roll from 3D landmarks).
- "Cheating" is never a direct model output — it's decided by a **scoring engine** you write on top of detection + tracking + head pose.
- Database: **Firebase Firestore only** (not Firebase Storage — as of Feb 2026 Storage requires linking a paid Blaze billing account, even for free-tier usage). Screenshots stay saved **locally** on disk (`alerts/screenshots/`), and Firestore just stores the local file path alongside each alert record — no card needed, and Firestore's Spark (free) plan has no time-based expiry, only a daily quota that resets automatically. A **local JSON fallback** is also used if Firestore credentials aren't set up yet (so the app never breaks while you're still developing).
- Video source: **both** file upload and live camera.
- Notifications: in-app popup (Tkinter `messagebox`), not an OS-level notification.
- Desktop app: **Tkinter**, with background-thread processing + a `Queue` so the UI never freezes.
- Screenshot Viewer is a required part of the UI (not optional).
- Final delivery includes a standalone `.exe` (Phase 19) with a custom icon, distributed via a GitHub Release — so anyone can download and double-click to run it, no Python or terminal needed.
- Development happens **fully locally** (no GPU needed — pretrained nano YOLO models run fine on CPU, and Tkinter/camera access require a local machine anyway; cloud notebooks like Kaggle/Colab are not used for this project).
- Automated unit tests are **optional**, placed near the end — do them only if you still have time/energy after everything else works.
- This is meant to be a flagship portfolio project, so a handful of "professional touches" (Phase 8) are included on purpose — they are simple to implement but are exactly what separates a hobby script from a project that reads as production-minded in a portfolio review.

**Coding style rule for every phase (because Ahmed is learning, not just shipping):**
- Every function/code block written for any phase must include comments explaining *why* it does something, not just *what* — e.g. not `# loop over boxes` but `# check every detected box to see if it's a phone near this student`.
- Every file should have clear section headers as comments (e.g. `# ---- Tracking logic ----`, `# ---- Scoring rules ----`) so it's easy to navigate even as files grow.
- When a phase is handled in a new chat, ask for an explanation of any new concept/library call the first time it's used, don't just paste code silently — the point is to understand it, not just to have it.

---

## 0. Project Overview & Folder Structure

**Status:** [ ] Not started

**Goal:** know exactly what the finished project looks like before writing any code.

**Final folder structure (grows into this gradually, phase by phase):**
```
exam_cheating_detection/
├── main.py                  # Tkinter GUI entry point
├── detector.py               # YOLO detection + tracking
├── head_pose.py               # MediaPipe head pose logic
├── scoring.py                   # Scoring engine (state dict + rules)
├── reliability.py                 # Frame-persistence + cooldown + background filtering
├── firebase_db.py                   # Firebase read/write + local JSON fallback
├── config.py                          # ALL tunable thresholds and paths live here
├── models/
│   └── yolov8n.pt                       # auto-downloaded by Ultralytics on first run
├── alerts/
│   ├── screenshots/                       # saved incident images
│   └── local_alerts.json                    # used when Firebase isn't configured
├── videos/                                    # processed output videos (optional to save)
├── tests/                                       # optional, Phase 17
│   └── test_scoring.py
├── requirements.txt
└── README.md
```

**Deliverable:** an empty repo with this folder structure created and `git init` done, so every later phase just fills in files that already have a home.


**Notes (fill in after finishing this phase):**
_(explain here, in your own words, what you built and anything that confused you or that you had to look up — this becomes your own reference later)_

---

## 1. Project Setup

**Status:** [ ] Not started

**Goal:** environment ready, nothing else.

**Steps:**
1. Create a virtual environment and install: `ultralytics`, `opencv-python`, `mediapipe`, `pillow`, `firebase-admin`.
2. Create `config.py` now, even mostly empty — you'll add values to it in every later phase instead of hardcoding numbers in the logic files:
```python
# config.py
MODEL_PATH = "models/yolov8n.pt"
CONF_THRESHOLD = 0.4
SCREENSHOT_DIR = "alerts/screenshots"
LOCAL_ALERTS_JSON = "alerts/local_alerts.json"
```
3. Create the folders from Phase 0 with placeholder `.gitkeep` files where needed.
4. Write a `requirements.txt` (`pip freeze > requirements.txt`).

**Deliverable:** a runnable empty project — `python -c "import ultralytics, cv2, mediapipe"` works with no errors.


**Notes (fill in after finishing this phase):**
_(explain here, in your own words, what you built and anything that confused you or that you had to look up — this becomes your own reference later)_

---

## 2. Basic Person Detection

**Status:** [ ] Not started

**Goal:** see YOLO detect people on a video, nothing more — no tracking, no logic yet.

**Steps:**
1. In `detector.py`, load the pretrained model and run it on a sample exam-like video, filtering only class `0` (person).
2. Draw a plain green box on every detected person.
3. Play the annotated video back (`cv2.imshow`) frame by frame just to confirm detection quality.

**Code hint:**
```python
from ultralytics import YOLO
import cv2
from config import MODEL_PATH, CONF_THRESHOLD

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture("sample_exam.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame, classes=[0], conf=CONF_THRESHOLD, verbose=False)
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Detection Test", frame)
    if cv2.waitKey(1) == ord('q'):
        break
```

**Deliverable:** a video window showing green boxes correctly following every student's body, no crashes.


**Notes (fill in after finishing this phase):**
_(explain here, in your own words, what you built and anything that confused you or that you had to look up — this becomes your own reference later)_

---

## 3. Multi-Object Tracking

**Status:** [ ] Not started

**Goal:** every student gets a stable ID that doesn't change from frame to frame.

**Steps:**
1. Replace `model.predict(...)` with `model.track(..., persist=True, tracker="botsort.yaml")` (or `bytetrack.yaml`).
2. Read `results[0].boxes.id` and draw the `track_id` as text above each box.
3. Watch the video and confirm the same student keeps the same number even if they move or get briefly occluded.

**Deliverable:** annotated video where each student has a consistent ID number the whole time (small ID flickering is normal for now — Phase 8 improves this).


**Notes (fill in after finishing this phase):**
_(explain here, in your own words, what you built and anything that confused you or that you had to look up — this becomes your own reference later)_

---

## 4. Phone Detection + Association Logic

**Status:** [ ] Not started

**Goal:** know *which* student a detected phone belongs to.

**Steps:**
1. Add class `67` (cell phone) to the same `model.track(classes=[0, 67], ...)` call.
2. Write a small function that expands a student's bounding box by ~15% and checks whether a phone box's center point falls inside it.
3. If yes, mark that `track_id` as "phone_detected = True" for this frame (don't trigger any alert yet — that's Phase 7).

**Code hint:**
```python
def expand_box(box, ratio=0.15):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (x1 - w*ratio, y1 - h*ratio, x2 + w*ratio, y2 + h*ratio)

def point_in_box(px, py, box):
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2
```

**Deliverable:** console print (or on-screen text) confirming "phone linked to student #3" whenever a phone appears near a tracked student.


**Notes (fill in after finishing this phase):**
_(explain here, in your own words, what you built and anything that confused you or that you had to look up — this becomes your own reference later)_

---

## 5. Head Pose Detection (standalone)

**Status:** [ ] Not started

**Goal:** get pitch/yaw/roll numbers from a single face, nothing tied to tracking yet.

**Steps:**
1. Run MediaPipe Face Mesh on a webcam feed or a face video by itself (separate small script, not integrated yet).
2. Extract the 3D landmarks needed for nose tip, chin, and eye corners.
3. Compute approximate pitch/yaw/roll and print them live.
4. Manually test: turn your head left/right/down and confirm the numbers change in the expected direction.

**Deliverable:** a standalone script printing head-pose angles that clearly change when you turn your head.


**Notes (fill in after finishing this phase):**
_(explain here, in your own words, what you built and anything that confused you or that you had to look up — this becomes your own reference later)_

---

## 6. Combine Head Pose with Tracked Students

**Status:** [ ] Not started

**Goal:** attach a head-pose reading to each tracked student, not just one face.

**Steps:**
1. For every tracked student box, crop that region and run Face Mesh only inside it (much faster than running it on the whole frame).
2. Store each student's current yaw/pitch in a small per-track dictionary (`current_head_pose[track_id] = {...}`).
3. If no face is found inside a student's box for that frame, mark `face_missing = True` for that `track_id`.

**Deliverable:** live video where each student's box also shows their current yaw/pitch numbers and a "face missing" flag when applicable.


**Notes (fill in after finishing this phase):**
_(explain here, in your own words, what you built and anything that confused you or that you had to look up — this becomes your own reference later)_

---

## 7. Scoring Engine

**Status:** [ ] Not started

**Goal:** turn frame-level signals (phone, head angle, face missing) into a time-based decision, so one bad frame never triggers a false alert.

**Steps:**
1. In `scoring.py`, create one dictionary keyed by `track_id`, storing: `score`, `looking_away_since`, `face_missing_since`.
2. Add these rules to `config.py` so they're easy to tune later:
```python
YAW_THRESHOLD_DEGREES = 20
LOOK_AWAY_SECONDS = 4
LOOK_AWAY_POINTS = 40
FACE_MISSING_SECONDS = 3
FACE_MISSING_POINTS = 50
PHONE_POINTS = 100
CHEATING_THRESHOLD = 80
```
3. Every frame, for every tracked student: update timers, add points when a rule's time condition is met, and check if `score >= CHEATING_THRESHOLD`.
4. When the threshold is crossed, this is the moment that triggers everything in Phase 9 onward (screenshot, DB write, alert) — but **only once per crossing**, not every frame the score stays above 80 (this connects directly to the cooldown logic in Phase 8).

**Deliverable:** console log showing a student's score climbing over time and a clear "ALERT: student #3 crossed threshold" message when it happens.


**Notes (fill in after finishing this phase):**
_(explain here, in your own words, what you built and anything that confused you or that you had to look up — this becomes your own reference later)_

---

## 8. Reliability Layer (Professional Touches)

**Status:** [ ] Not started

**Goal:** this phase is what makes the project feel production-grade instead of a demo script — and it's still simple to write.

**Steps:**
1. **Frame-persistence confirmation:** don't fire an alert from a single frame's phone detection — require the same behavior to be true for `CONFIRM_FRAMES` (e.g. 3) consecutive frames before it counts, and require it to be *absent* for the same number of frames before clearing. This alone removes most false positives (a pen misread as a phone for one frame, a tracker glitch, etc.).
2. **Background-person filtering:** ignore any detected person whose box height is below a ratio of the *tallest* detected person in the frame (e.g. `RELATIVE_MIN_HEIGHT_RATIO = 0.6`) — this filters out people walking in the background who aren't seated at an exam desk.
3. **Alert cooldown:** once a student triggers an alert, don't let the same student trigger another one for `ALERT_COOLDOWN_SECONDS` (e.g. 10s) — prevents the same incident from spamming ten alerts in two seconds.
4. Add all of these as named constants in `config.py`, not hardcoded numbers, so you can tune sensitivity without touching logic files.

**Deliverable:** a short before/after test — run the same video with and without this phase's logic and count how many false alerts each version produces. This comparison itself is great material for your README later.


**Notes (fill in after finishing this phase):**
_(explain here, in your own words, what you built and anything that confused you or that you had to look up — this becomes your own reference later)_

---

## 9. Screenshot Capture on Alert

**Status:** [ ] Not started

**Goal:** save visual evidence the moment an alert fires — locally first, Firebase comes next.

**Steps:**
1. When Phase 7+8 confirms a real alert, crop (or capture the full frame) and save it to `alerts/screenshots/` with a filename like `track3_20260912_143501.jpg`.
2. Store the file path in memory for now (DB write comes in Phase 11).

**Deliverable:** a growing `alerts/screenshots/` folder with correctly named, correctly cropped images after running on a test video.


**Notes (fill in after finishing this phase):**
_(explain here, in your own words, what you built and anything that confused you or that you had to look up — this becomes your own reference later)_

---

## 10. Firebase Setup

**Status:** [ ] Not started

**Goal:** get your Firebase project ready — no application code yet, just the account/console side.

**Steps:**
1. Create a Firebase project in the console.
2. Enable **Firestore Database** only (skip Firebase Storage — since Feb 2026 it requires a linked Blaze/billing account even for free-tier usage, and we don't need it: screenshots stay local on disk, only their file path goes into Firestore).
3. Go to Project Settings → Service Accounts → Generate new private key → save as `firebase_credentials.json` in the project root (and add it to `.gitignore` immediately — never commit this file).

**Deliverable:** a working Firebase project with Firestore enabled (Spark/free plan, no card needed) and a credentials file saved locally (never pushed to GitHub).


**Notes (fill in after finishing this phase):**
_(explain here, in your own words, what you built and anything that confused you or that you had to look up — this becomes your own reference later)_

---

## 11. Firebase Integration + Local Fallback

**Status:** [ ] Not started

**Goal:** every alert gets logged reliably, whether or not Firebase is reachable.

**Steps:**
1. In `firebase_db.py`, write two functions: `add_alert(record)` and `get_all_alerts()`.
2. Inside `add_alert`, try to: write a Firestore document with (timestamp, track_id, behavior, confidence, **local** `screenshot_path`) — the image itself never leaves your disk, only its path is logged.
3. Wrap this in a `try/except` — if it fails (no credentials, no internet), append the same record as a line in `alerts/local_alerts.json` instead, and print a notice instead of crashing.
4. Keep the function **signatures** identical regardless of which backend actually wrote the data — this means your Tkinter code (Phase 14) never needs to know or care which one was used.

**Deliverable:** run the app with Firebase configured → data appears in the Firebase console. Rename the credentials file temporarily → app still runs and logs to `local_alerts.json` instead of crashing.


**Notes (fill in after finishing this phase):**
_(explain here, in your own words, what you built and anything that confused you or that you had to look up — this becomes your own reference later)_

---

## 12. Basic Tkinter UI

**Status:** [ ] Not started

**Goal:** the simplest possible working window — just video in, video displayed. No alerts panel yet.

**Steps:**
1. Build a Tkinter window with an "Upload Video" button and a `Label` widget used as a video canvas.
2. On button click, open a file dialog, then run detection+tracking (Phases 2–3 only, for now) on the selected video and display each frame in the label using Pillow (`ImageTk.PhotoImage`).

**Deliverable:** a window where you upload a video and watch the green/tracked boxes play back live inside the app (not in a separate OpenCV window anymore).


**Notes (fill in after finishing this phase):**
_(explain here, in your own words, what you built and anything that confused you or that you had to look up — this becomes your own reference later)_

---

## 13. Live Camera Support

**Status:** [ ] Not started

**Goal:** add the second video source next to file upload.

**Steps:**
1. Add a "Start Camera" button that opens `cv2.VideoCapture(0)` instead of a file, feeding the same processing loop from Phase 12.
2. Add a "Stop" button that cleanly releases whichever source (file or camera) is currently running.

**Deliverable:** the same live-preview behavior works identically whether the source is an uploaded file or the webcam.


**Notes (fill in after finishing this phase):**
_(explain here, in your own words, what you built and anything that confused you or that you had to look up — this becomes your own reference later)_

---

## 14. Alerts Panel (UI)

**Status:** [ ] Not started

**Goal:** the part of the UI that makes this feel like a real monitoring tool.

**Steps:**
1. Add a `Treeview` table below the video canvas showing: timestamp, student ID, behavior, confidence, screenshot path.
2. Every time `add_alert()` (Phase 11) is called, also insert a new row into this table and pop up a `messagebox.showwarning(...)` with the incident summary.
3. Add a "View Screenshot" button (or double-click a row) that opens the saved image in a new window.

**Deliverable:** running a test video produces live rows in the table, a popup per new incident, and a working screenshot viewer on demand.


**Notes (fill in after finishing this phase):**
_(explain here, in your own words, what you built and anything that confused you or that you had to look up — this becomes your own reference later)_

---

## 15. Threading Integration

**Status:** [ ] Not started

**Goal:** stop the UI from freezing while heavy detection runs.

**Steps:**
1. Move the entire detection/tracking/scoring loop (Phases 2–11) into a background `threading.Thread`.
2. Use a `queue.Queue` to pass finished frames and new alerts from that thread back to the main Tkinter thread.
3. In the main thread, use `root.after(...)` to poll the queue every ~15ms and update the video label / alerts table from there — never touch Tkinter widgets directly from the background thread.

**Deliverable:** the app stays fully responsive (buttons clickable, window draggable) while a video is being processed in the background.


**Notes (fill in after finishing this phase):**
_(explain here, in your own words, what you built and anything that confused you or that you had to look up — this becomes your own reference later)_

---

## 16. Testing & Threshold Tuning

**Status:** [ ] Not started

**Goal:** run the finished system against a real (or realistic) exam video and tune the numbers in `config.py` until it behaves well.

**Steps:**
1. Record or find a short mock exam video (a few people, someone briefly checking a phone, someone looking around).
2. Run the full pipeline and note every false positive and every missed real incident.
3. Adjust `config.py` values (thresholds, `CONFIRM_FRAMES`, `ALERT_COOLDOWN_SECONDS`) one at a time and re-test — never change more than one value between test runs, or you won't know which change fixed (or broke) anything.

**Deliverable:** a short written note (a paragraph is enough) listing what you changed and why — this becomes useful material for your README's "design decisions" section.


**Notes (fill in after finishing this phase):**
_(explain here, in your own words, what you built and anything that confused you or that you had to look up — this becomes your own reference later)_

---

## 17. [Optional] Automated Unit Tests

**Status:** [ ] Not started

**Goal:** only do this phase if you have time/energy left after everything above works. Skip it entirely without guilt if not — it does not block anything else.

**Steps:**
1. Install `pytest`.
2. Write small tests only for pure-logic functions that don't need a camera or a real video — mainly `scoring.py` (does each rule add the right points, does crossing 80 trigger correctly) and the bbox-association function from Phase 4.
3. Aim for roughly 15–20 focused tests, not exhaustive coverage — the goal is "I can prove my core logic is correct," not "every line is covered."

**Deliverable:** a `tests/` folder that runs green with `pytest`, and one line in your README: "core scoring and detection-association logic is covered by unit tests."


**Notes (fill in after finishing this phase):**
_(explain here, in your own words, what you built and anything that confused you or that you had to look up — this becomes your own reference later)_

---

## 18. Portfolio Packaging

**Status:** [ ] Not started

**Goal:** make this read as your flagship project on GitHub/LinkedIn.

**Steps:**
1. Write a `README.md` with: problem statement, how the scoring engine works (this is your most original part — explain it well), the reliability techniques from Phase 8, tech stack, screenshots of the UI, and how to run it.
2. Explicitly state the ethical framing: the system produces **evidence for human review**, not an automatic disciplinary decision — no face recognition, no stored student names, just anonymous track IDs. This single paragraph matters a lot for how professionally the project reads.
3. Record a short screen-capture GIF/video showing: upload → live detection → an alert firing → the screenshot viewer, and embed it in the README.
4. Push the code (with `firebase_credentials.json` excluded via `.gitignore`), and write a short LinkedIn post summarizing the project and linking the repo.

**Deliverable:** a public GitHub repo with a strong README, demo media, and (if Phase 17 was done) a visible "tested" badge/mention — ready to link from your portfolio.


**Notes (fill in after finishing this phase):**
_(explain here, in your own words, what you built and anything that confused you or that you had to look up — this becomes your own reference later)_

---

## 19. Standalone Executable & Easy Distribution

**Status:** [ ] Not started

**Goal:** make the app feel like a real professional program — someone downloads it from GitHub and just double-clicks an icon to run it, with no Python installation, no `pip install`, no terminal at all.

**Steps:**
1. Install PyInstaller: `pip install pyinstaller`.
2. Get (or make) a simple `.ico` file and save it as `assets/icon.ico` — any free PNG-to-ICO converter online works if you design a quick logo/icon.
3. Make sure any file your code loads by a relative path (the YOLO weights, the icon, `firebase_credentials.json`) is looked up in a way that still works once everything is bundled — PyInstaller extracts bundled files to a temporary folder at runtime, so use a small helper like this instead of a plain path string:
```python
import sys, os

def resource_path(relative_path):
    # when running as a normal .py file, use the current folder
    # when running as a PyInstaller .exe, files are unpacked into a temp folder (sys._MEIPASS)
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, relative_path)
```
4. Build the executable (Windows example):
```bash
pyinstaller --onefile --windowed --icon=assets/icon.ico --name "ExamCheatingDetector" main.py
```
   - `--onefile` → everything packed into a single `.exe`.
   - `--windowed` → no black console window pops up behind your Tkinter GUI.
   - `--icon` → gives the `.exe` your custom icon, shown in Windows Explorer too.
5. Test the generated `.exe` (found in the `dist/` folder) on its own — close your code editor and just double-click it, to confirm it truly runs standalone.
6. On GitHub, don't commit the `.exe` to the repo itself (binaries bloat git history) — instead create a **GitHub Release**, attach the `.exe` there as a downloadable asset, and link to it from your README with a "⬇️ Download for Windows" badge/button.
7. Keep the source-code path available too for developers who want to read/modify the code: `git clone` → `pip install -r requirements.txt` → `python main.py`. Your README should offer both: "just want to run it? Download the .exe from Releases. Want to see/change the code? Clone and run with Python."

**Deliverable:** a single `ExamCheatingDetector.exe` with your custom icon that a complete stranger can download from your GitHub Releases page and double-click to run — no terminal, no Python, no setup.

**Notes (fill in after finishing this phase):**
_(explain here, in your own words, what you built and anything that confused you or that you had to look up — this becomes your own reference later)_

---

## How to use this document with me

Send me **one numbered phase at a time** (e.g. "let's do Phase 4" or paste that section directly) in a new message, and I'll pick up with full context — no need to re-explain the project each time. Mention which phase you're on and I'll go straight to code/execution for that step.
