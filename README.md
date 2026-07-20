# MindSync 🧠
### An Attention-Aware Intelligent Learning System Using Behavioural and Visual Cues

MindSync is a real-time cognitive state detection system that monitors a learner's attention, fatigue, and confusion through webcam input and adapts learning content accordingly — simulating the responsiveness of a human tutor.

---

## Project Status
> B.Tech Final Year Project — currently in active development (2nd Year)

| Module | Status |
|--------|--------|
| Environment Setup | ✅ Complete |
| Webcam Capture | ✅ Complete |
| Face Mesh (468 landmarks) | ✅ Complete |
| Eye Landmark Extraction | ✅ Complete |
| EAR + Blink/Fatigue Detection | ✅ Complete |
| Head Pose Estimation | ✅ Complete |
| Cognitive State Classifier | ✅ Complete |
| Adaptive Learning Engine | ✅ Complete |
| Flask Backend | 🔲 Upcoming |
| React Frontend + Dashboard | 🔲 Upcoming |

---

## What It Does

MindSync continuously monitors the learner through their webcam and classifies their cognitive state in real time into one of four categories:

| State | Detection Method | Adaptive Response |
|-------|-----------------|-------------------|
| **Focused** | Normal EAR + straight head pose | Increase content difficulty |
| **Distracted** | Head turned away (yaw/pitch > threshold) | Trigger attention alert |
| **Fatigued** | EAR below threshold for 30+ frames | Suggest a break |
| **Confused** | Frequent blinks + mild head movement | Simplify content |

---

## System Architecture

```
Webcam Input
     │
     ▼
Face Mesh (MediaPipe)
     │
     ├──► Eye Landmark Extraction
     │         └──► EAR Calculation ──► Fatigue / Blink Detection
     │
     └──► Head Pose Estimation ──► Distraction Detection
               │
               ▼
        Cognitive State Classifier
        (Focused / Distracted / Fatigued / Confused)
               │
               ▼
        Adaptive Learning Engine
        (Action Recommendations + Session Logging)
```

---

## Project Structure

```
MindSync/
│
├── modules/
│   ├── data_acquisition/
│   │   └── capture.py               # Webcam feed with FPS display
│   ├── cognitive_detection/
│   │   ├── face_mesh.py             # MediaPipe face landmark detection
│   │   ├── eye_extractor.py         # Eye landmark extraction (6 points/eye)
│   │   ├── ear_calculator.py        # EAR computation + blink counting
│   │   ├── head_pose.py             # solvePnP head pose estimation
│   │   └── state_classifier.py      # Unified cognitive state classifier
│   ├── adaptive_engine/
│   │   ├── engine.py                # Adaptive recommendation engine
│   │   └── runner.py                # Integrated detection + adaptation runner
│   └── analytics/                   # (Upcoming) Session analytics
│
├── models/
│   └── face_landmarker.task         # MediaPipe face landmark model
│
├── data/
│   ├── raw/                         # Raw session recordings
│   └── processed/                   # Extracted feature data
│
├── static/                          # Frontend assets (upcoming)
├── templates/                       # Flask HTML templates (upcoming)
├── tests/                           # Unit tests
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.13 |
| Computer Vision | OpenCV 4.x |
| Face Landmark Detection | MediaPipe 0.10.x (Tasks API) |
| ML / Classification | Rule-based (Scikit-learn upcoming) |
| Backend | Flask (upcoming) |
| Frontend | React (upcoming) |
| Version Control | Git / GitHub |
| IDE | VS Code |

---

## Getting Started

### Prerequisites
- Python 3.13
- Webcam

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/MindSync.git
cd MindSync

# Create and activate virtual environment
python -m venv mindsync-env
mindsync-env\Scripts\activate   # Windows
source mindsync-env/bin/activate # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Download MediaPipe face landmark model
python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', 'models/face_landmarker.task'); print('Downloaded!')"
```

### Running the System

```bash
# Run the full adaptive learning system
python modules/adaptive_engine/runner.py

# Run individual modules
python modules/data_acquisition/capture.py          # Webcam feed
python modules/cognitive_detection/face_mesh.py     # Face mesh
python modules/cognitive_detection/ear_calculator.py # EAR detection
python modules/cognitive_detection/head_pose.py     # Head pose
python modules/cognitive_detection/state_classifier.py # Full classifier
```

---

## Key Technical Concepts

### Eye Aspect Ratio (EAR)
```
EAR = (||P2-P6|| + ||P3-P5||) / (2 × ||P1-P4||)

        P2   P3
P1  .    .   .   . P4
        P5   P6

EAR > 0.20  → Eye open  (Focused)
EAR < 0.20  → Eye closed
Closed for 30+ frames → Fatigued
```

### Head Pose Estimation
Using `cv2.solvePnP` with 6 facial anchor points to compute:
- **Yaw** — left/right rotation (distraction detection)
- **Pitch** — up/down tilt (looking away detection)
- **Roll** — sideways tilt

Calibration offsets are applied to account for webcam angle and positioning.

---

## Known Limitations

- Accuracy depends on lighting and webcam quality
- Calibration offsets are currently hardcoded per device
- Limited to single-user sessions
- Privacy: requires webcam access

---

## Future Scope

- [ ] Flask REST API backend
- [ ] React frontend with learning interface
- [ ] Analytics dashboard (attention trends, session reports)
- [ ] ML-based classifier trained on real session data (Scikit-learn)
- [ ] EEG / wearable device integration
- [ ] Multi-user support
- [ ] SaaS deployment

---

## Author

**Sreetama Santra**  
B.Tech Student | Computer Science & Engineering 

---

## License
This project is for academic purposes as part of a B.Tech Final Year Project.
