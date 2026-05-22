import cv2
import mediapipe as mp
import numpy as np
import time

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Eye landmarks
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Head pose landmarks
POSE_LANDMARK_INDICES = [1, 152, 263, 33, 287, 57]
FACE_3D_POINTS = np.array([
    [0.0,    0.0,    0.0   ],
    [0.0,   -63.6,  -12.5 ],
    [-43.3,  32.7,  -26.0 ],
    [43.3,   32.7,  -26.0 ],
    [-28.9, -28.9,  -24.1 ],
    [28.9,  -28.9,  -24.1 ]
], dtype=np.float64)

# Calibration offsets (from webcam)
PITCH_OFFSET = 15
YAW_OFFSET   = -12

# Thresholds
EAR_THRESHOLD      = 0.20
FATIGUE_FRAMES     = 30
BLINK_FRAMES       = 3
DISTRACTION_ANGLE  = 25
CONFUSION_ANGLE    = 10
CONFUSION_BLINKS   = 3
CONFUSION_WINDOW   = 5  # seconds

# State colors
STATE_COLORS = {
    "Focused":    (0,   255, 0  ),
    "Distracted": (0,   165, 255),
    "Fatigued":   (0,   0,   255),
    "Confused":   (255, 0,   255)
}

# helpers

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(points):
    v1  = euclidean(points[1], points[5])
    v2  = euclidean(points[2], points[4])
    h   = euclidean(points[0], points[3])
    return (v1 + v2) / (2.0 * h)

def get_eye_points(landmarks, indices, w, h):
    return [(int(landmarks[i].x * w),
             int(landmarks[i].y * h)) for i in indices]

def get_head_pose(landmarks, fw, fh):
    image_points = np.array(
        [[landmarks[i].x * fw, landmarks[i].y * fh]
         for i in POSE_LANDMARK_INDICES], dtype=np.float64)

    camera_matrix = np.array([
        [fw, 0,      fw / 2],
        [0,  fw,     fh / 2],
        [0,  0,      1     ]
    ], dtype=np.float64)

    _, rvec, _ = cv2.solvePnP(
        FACE_3D_POINTS, image_points,
        camera_matrix, np.zeros((4,1)),
        flags=cv2.SOLVEPNP_ITERATIVE)

    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)

    if sy > 1e-6:
        pitch = np.degrees(np.arctan2( rmat[2,1], rmat[2,2]))
        yaw   = np.degrees(np.arctan2(-rmat[2,0], sy))
    else:
        pitch = np.degrees(np.arctan2(-rmat[1,2], rmat[1,1]))
        yaw   = np.degrees(np.arctan2(-rmat[2,0], sy))

    return round(pitch, 1), round(yaw, 1)

def classify_state(avg_ear, pitch_adj, yaw_adj,
                   closed_frames, recent_blinks):
    """
    Priority: Fatigued > Distracted > Confused > Focused
    """
    if closed_frames >= FATIGUE_FRAMES:
        return "Fatigued"

    if abs(yaw_adj) > DISTRACTION_ANGLE or abs(pitch_adj) > DISTRACTION_ANGLE:
        return "Distracted"

    if (recent_blinks >= CONFUSION_BLINKS and
            (abs(yaw_adj) > CONFUSION_ANGLE or
             abs(pitch_adj) > CONFUSION_ANGLE)):
        return "Confused"

    return "Focused"

# main 

def start_classifier():
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path='models/face_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1)

    cap = cv2.VideoCapture(0)

    closed_frames  = 0
    blink_count    = 0
    blink_times    = []   # timestamps of each blink
    prev_ear_low   = False

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            fh, fw, _ = frame.shape
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results  = landmarker.detect(mp_image)

            if results.face_landmarks:
                lm = results.face_landmarks[0]

                # EAR
                left_pts  = get_eye_points(lm, LEFT_EYE,  fw, fh)
                right_pts = get_eye_points(lm, RIGHT_EYE, fw, fh)
                avg_ear   = (calculate_ear(left_pts) +
                             calculate_ear(right_pts)) / 2.0

                # Blink detection 
                if avg_ear < EAR_THRESHOLD:
                    closed_frames += 1
                    prev_ear_low   = True
                else:
                    if prev_ear_low:          # eye just opened → blink
                        if closed_frames <= BLINK_FRAMES * 3:
                            blink_count += 1
                            blink_times.append(time.time())
                    closed_frames = 0
                    prev_ear_low  = False

                # Keep only blinks in last CONFUSION_WINDOW seconds
                now          = time.time()
                blink_times  = [t for t in blink_times
                                if now - t <= CONFUSION_WINDOW]
                recent_blinks = len(blink_times)

                # Head pose
                pitch, yaw   = get_head_pose(lm, fw, fh)
                pitch_adj    = pitch - PITCH_OFFSET
                yaw_adj      = yaw   - YAW_OFFSET

                # Classify
                state = classify_state(
                    avg_ear, pitch_adj, yaw_adj,
                    closed_frames, recent_blinks)
                color = STATE_COLORS[state]

                # Display
                cv2.putText(frame,
                    f"EAR: {round(avg_ear,3)}",       (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                cv2.putText(frame,
                    f"Blinks (5s): {recent_blinks}",  (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                cv2.putText(frame,
                    f"Pitch adj: {round(pitch_adj,1)}",(10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                cv2.putText(frame,
                    f"Yaw adj:   {round(yaw_adj,1)}",  (10,105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

                # Big state label
                cv2.putText(frame,
                    f"STATE: {state}",                (10,145),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

                # Draw eye points
                for p in left_pts + right_pts:
                    cv2.circle(frame, p, 2, color, -1)

            else:
                cv2.putText(frame, "No Face Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            cv2.imshow("MindSync - Cognitive State", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_classifier()
    