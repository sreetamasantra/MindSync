import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Thresholds
EAR_THRESHOLD    = 0.20   # Below this = eye closed
FATIGUE_FRAMES   = 30     # Frames eye must stay closed to flag fatigue
BLINK_FRAMES     = 3      # Short closure = blink

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(eye_points):
    # Vertical distances
    v1 = euclidean(eye_points[1], eye_points[5])
    v2 = euclidean(eye_points[2], eye_points[4])
    # Horizontal distance
    h  = euclidean(eye_points[0], eye_points[3])
    ear = (v1 + v2) / (2.0 * h)
    return round(ear, 3)

def get_eye_points(landmarks, indices, w, h):
    return [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]

def start_ear():
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='models/face_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1
    )

    cap = cv2.VideoCapture(0)

    closed_frames = 0
    blink_count   = 0
    state         = "Focused"

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            fh, fw, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = landmarker.detect(mp_image)

            if results.face_landmarks:
                lm = results.face_landmarks[0]

                left_pts  = get_eye_points(lm, LEFT_EYE,  fw, fh)
                right_pts = get_eye_points(lm, RIGHT_EYE, fw, fh)

                left_ear  = calculate_ear(left_pts)
                right_ear = calculate_ear(right_pts)
                avg_ear   = round((left_ear + right_ear) / 2.0, 3)

                # State logic
                if avg_ear < EAR_THRESHOLD:
                    closed_frames += 1
                    if closed_frames == BLINK_FRAMES:
                        blink_count += 1
                    if closed_frames >= FATIGUE_FRAMES:
                        state = "Fatigued"
                else:
                    closed_frames = 0
                    state = "Focused"

                # Choose colour based on state
                color = (0, 255, 0) if state == "Focused" else (0, 0, 255)

                # Display info
                cv2.putText(frame, f"EAR: {avg_ear}",        (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Blinks: {blink_count}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"State: {state}",        (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Draw eye points
                for p in left_pts + right_pts:
                    cv2.circle(frame, p, 2, color, -1)

            else:
                cv2.putText(frame, "No Face Detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("MindSync - EAR Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_ear() 