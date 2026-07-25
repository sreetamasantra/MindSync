# MindSync 
### An Attention-Aware Intelligent Learning System Using Behavioural and Visual Cues

MindSync is a real-time cognitive state detection system that monitors a learner's attention, fatigue, and confusion through webcam input and adapts learning content accordingly — simulating the responsiveness of a human tutor.

---

## Project Status
> B.Tech Final Year Project — Core prototype complete

| Module | Status |
|--------|--------|
| Environment Setup | ✅ Complete |
| Webcam Capture (54 FPS, 640x480) | ✅ Complete |
| Face Mesh — 468 landmarks (MediaPipe) | ✅ Complete |
| Eye Landmark Extraction | ✅ Complete |
| EAR + Blink/Fatigue Detection | ✅ Complete |
| Head Pose Estimation (solvePnP) | ✅ Complete |
| Cognitive State Classifier | ✅ Complete |
| Adaptive Learning Engine | ✅ Complete |
| Flask REST API Backend | ✅ Complete |
| Labeled Dataset Collection (3921 samples) | ✅ Complete |
| Random Forest ML Classifier (85.73% accuracy) | ✅ Complete |
| ML Model Integrated into Live System | ✅ Complete |
| React.js Frontend | ✅ Complete (separate repo) |
| Analytics Dashboard | ✅ Complete (separate repo) |
| User Study | 🔲 Upcoming |

---

## What It Does

MindSync continuously monitors the learner through their webcam and classifies their cognitive state in real time into one of four categories:

| State | Detection Method | Adaptive Response |
|-------|-----------------|-------------------|
| **Focused** | Normal EAR + straight head pose | Increase content difficulty |
| **Distracted** | Head turned away (yaw/pitch > threshold) | Trigger attention alert |
| **Fatigued** | EAR below threshold for sustained frames | Suggest a break |
| **Confused** | Frequent blinks + mild head movement | Simplify content |

---

## ML Model Results

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest (100 estimators) |
| Dataset | 3921 labeled samples (self-collected) |
| Train/Test Split | 80% / 20% |
| **Accuracy** | **85.73%** |
| **Weighted F1-Score** | **85.78%** |
| Cross-Validation (5-fold) | 74.16% (± 4.98%) |

### Per-Class Performance:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Confused | 0.76 | 0.90 | 0.82 | 181 |
| Distracted | 0.92 | 0.82 | 0.87 | 184 |
| Fatigued | 0.90 | 0.92 | 0.91 | 215 |
| Focused | 0.86 | 0.80 | 0.83 | 205 |

### Feature Importances:

| Feature | Importance |
|---------|-----------|
| pitch_adj | 0.2968 |
| blink_rate | 0.2408 |
| yaw_adj | 0.2261 |
| ear | 0.1405 |
| closed_frames | 0.0957 |

---

## System Architecture

```
Webcam Input
     │
     ▼
Face Mesh (MediaPipe — 468 landmarks)
     │
     ├──► Eye Landmark Extraction (6 points/eye)
     │         └──► EAR Calculation ──► Fatigue / Blink Detection
     │
     └──► Head Pose Estimation (solvePnP)
               └──► Yaw / Pitch ──► Distraction Detection
                         │
                         ▼
                Cognitive State Classifier
                (Random Forest ML Model)
                (Focused / Distracted / Fatigued / Confused)
                         │
                         ▼
                Adaptive Learning Engine
                (Action Recommendations + Session Logging)
                         │
                         ▼
                Flask REST API
                (/api/state, /api/session/log, /api/session/reset)
                         │
                         ▼
                React.js Frontend
                (Live UI + Analytics Dashboard)
```

---

## Project Structure

```
MindSync/
│
├── modules/
│   ├── data_acquisition/
│   │   ├── capture.py                  # Webcam feed with FPS display
│   │   └── collect_data.py             # Labeled data collection pipeline
│   ├── cognitive_detection/
│   │   ├── face_mesh.py                # MediaPipe face landmark detection
│   │   ├── eye_extractor.py            # Eye landmark extraction
│   │   ├── ear_calculator.py           # EAR computation + blink counting
│   │   ├── head_pose.py                # Head pose estimation
│   │   ├── state_classifier.py         # Rule-based classifier (base)
│   │   ├── ml_classifier.py            # ML model inference wrapper
│   │   └── train_model.py              # Model training + evaluation script
│   ├── adaptive_engine/
│   │   ├── engine.py                   # Adaptive recommendation engine
│   │   └── runner.py                   # Standalone runner
│   └── analytics/
│       ├── session_state.py            # Thread-safe shared state manager
│       └── detection_thread.py         # Background detection thread
│
├── models/
│   ├── face_landmarker.task            # MediaPipe face landmark model
│   ├── cognitive_classifier.pkl        # Trained Random Forest model
│   └── label_encoder.pkl              # Sklearn label encoder
│
├── data/
│   ├── raw/
│   │   └── session_data.csv            # Labeled dataset (3921 samples)
│   └── processed/
│       └── confusion_matrix.png        # Model evaluation visualization
│
├── app.py                              # Flask REST API server
├── requirements.txt
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/state` | Current cognitive state + metrics |
| GET | `/api/session/log` | Full session action log |
| POST | `/api/session/reset` | Reset the current session |

### Sample `/api/state` response:
```json
{
  "state": "Focused",
  "ear": 0.293,
  "pitch_adj": -2.1,
  "yaw_adj": 4.6,
  "blink_count": 12,
  "recent_blinks": 0,
  "confidence": 0.69,
  "recommendation": {
    "action": "increase_difficulty",
    "message": "Great focus! Increasing content difficulty."
  }
}
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.13 |
| Computer Vision | OpenCV 4.x |
| Face Landmark Detection | MediaPipe 0.10.x (Tasks API) |
| ML Model | Random Forest (Scikit-learn) |
| Backend | Flask + Flask-CORS |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
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
git clone https://github.com/sreetamasantra/MindSync.git
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

### Running the system

```bash
# Start the Flask backend (detection + API)
python app.py
```

API available at `http://localhost:5000`.
Run the frontend separately — see [MindSync-Frontend](https://github.com/sreetamasantra/MindSync-Frontend).

### Running individual modules

```bash
python modules/data_acquisition/capture.py             # Webcam feed
python modules/cognitive_detection/face_mesh.py        # Face mesh
python modules/cognitive_detection/ear_calculator.py   # EAR detection
python modules/cognitive_detection/head_pose.py        # Head pose
python modules/cognitive_detection/state_classifier.py # Full classifier
python modules/data_acquisition/collect_data.py        # Data collection
python modules/cognitive_detection/train_model.py      # Train ML model
```

---

## Key Technical Concepts

### Eye Aspect Ratio (EAR)
```
EAR = (||P2-P6|| + ||P3-P5||) / (2 × ||P1-P4||)

        P2   P3
P1  .    .   .   . P4
        P5   P6

EAR > 0.20  → Eye open
EAR < 0.20  → Eye closed / blinking
Closed for 30+ frames → Fatigued
```

### Head Pose Estimation
Using `cv2.solvePnP` with 6 facial anchor points mapped to a 3D face model to extract yaw and pitch angles. Calibration offsets applied to correct for webcam positioning.

### ML Pipeline
```
Features: EAR, pitch_adj, yaw_adj, blink_rate, closed_frames
     │
     ▼
Random Forest Classifier (100 trees, max_depth=10, balanced class weights)
     │
     ▼
Predicted cognitive state + confidence score
```

---

## Known Limitations

- Calibration offsets are hardcoded for a specific webcam setup
- Dataset collected from a single user — generalization may vary
- Accuracy depends on lighting conditions
- Privacy: requires continuous webcam access

---

## Future Scope

- [ ] User study with multiple participants
- [ ] Multi-user dataset for better generalization
- [ ] Deep learning / LSTM-based classifier
- [ ] Session persistence with database
- [ ] EEG / wearable device integration
- [ ] SaaS deployment

---

## Related Repository

Frontend (React.js): [MindSync-Frontend](https://github.com/sreetamasantra/MindSync-Frontend)

---

## Author

**Sreetama Santra**
B.Tech Student | CSE (IoT)


---

## 📄 License

This project is for academic purposes as part of a B.Tech Final Year Project.
