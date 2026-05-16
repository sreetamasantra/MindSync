import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 6 key landmark indices for each eye (for EAR calculation)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def get_eye_points(face_landmarks, eye_indices, frame_w, frame_h): 
    """Extract pixel coordinates for given eye landmark indices."""
    points = []
    for idx in eye_indices:
        lm = face_landmarks[idx]
        x = int(lm.x * frame_w)
        y = int(lm.y * frame_h)
        points.append((x, y))
    return points

def draw_eye_points(frame, points, color):
    """Draw circles on each eye landmark point."""
    for (x, y) in points:
        cv2.circle(frame, (x, y), 3, color, -1)

def start_eye_extraction():
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='models/face_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1
    )

    cap = cv2.VideoCapture(0)

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = landmarker.detect(mp_image)

            if results.face_landmarks:
                landmarks = results.face_landmarks[0]

                # Extract eye points
                left_points = get_eye_points(landmarks, LEFT_EYE, w, h)
                right_points = get_eye_points(landmarks, RIGHT_EYE, w, h)

                # Draw them
                draw_eye_points(frame, left_points, (0, 255, 0))   # Green = left
                draw_eye_points(frame, right_points, (255, 0, 0))  # Blue = right

                # Label each point with its index
                for i, (x, y) in enumerate(left_points):
                    cv2.putText(frame, f"L{i+1}", (x+4, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                for i, (x, y) in enumerate(right_points):
                    cv2.putText(frame, f"R{i+1}", (x+4, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

                cv2.putText(frame, "Eye landmarks extracted", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            else:
                cv2.putText(frame, "No Face Detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("MindSync - Eye Extraction", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_eye_extraction()