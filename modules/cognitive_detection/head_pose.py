import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

POSE_LANDMARK_INDICES = [1, 152, 263, 33, 287, 57]

FACE_3D_POINTS = np.array([
    [0.0,    0.0,    0.0   ],
    [0.0,   -63.6,  -12.5 ],
    [-43.3,  32.7,  -26.0 ],
    [43.3,   32.7,  -26.0 ],
    [-28.9, -28.9,  -24.1 ],
    [28.9,  -28.9,  -24.1 ]
], dtype=np.float64)

# Baseline offset calibrated from webcam
PITCH_OFFSET = 15
YAW_OFFSET   = -12

def get_head_pose(landmarks, frame_w, frame_h):
    image_points = []
    for idx in POSE_LANDMARK_INDICES:
        lm = landmarks[idx]
        image_points.append([lm.x * frame_w, lm.y * frame_h])
    image_points = np.array(image_points, dtype=np.float64)

    focal_length = frame_w
    camera_matrix = np.array([
        [focal_length, 0,            frame_w / 2],
        [0,            focal_length, frame_h / 2],
        [0,            0,            1          ]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, _ = cv2.solvePnP(
        FACE_3D_POINTS, image_points,
        camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)

    sy = np.sqrt(rotation_mat[0,0]**2 + rotation_mat[1,0]**2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2( rotation_mat[2,1], rotation_mat[2,2])
        yaw   = np.arctan2(-rotation_mat[2,0], sy)
        roll  = np.arctan2( rotation_mat[1,0], rotation_mat[0,0])
    else:
        pitch = np.arctan2(-rotation_mat[1,2], rotation_mat[1,1])
        yaw   = np.arctan2(-rotation_mat[2,0], sy)
        roll  = 0

    pitch = np.degrees(pitch)
    yaw   = np.degrees(yaw)
    roll  = np.degrees(roll)

    return round(pitch, 1), round(yaw, 1), round(roll, 1)

def start_head_pose():
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

            fh, fw, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = landmarker.detect(mp_image)

            if results.face_landmarks:
                lm = results.face_landmarks[0]
                pitch, yaw, roll = get_head_pose(lm, fw, fh)

                # Apply calibration offset
                pitch_adj = pitch - PITCH_OFFSET
                yaw_adj   = yaw   - YAW_OFFSET

                # Determine state
                if abs(yaw_adj) > 25 or abs(pitch_adj) > 25:
                    state = "Distracted"
                    color = (0, 0, 255)
                else:
                    state = "Focused"
                    color = (0, 255, 0)

                cv2.putText(frame, f"Pitch: {pitch} (adj: {round(pitch_adj,1)})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Yaw:   {yaw} (adj: {round(yaw_adj,1)})",     (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Roll:  {roll}",                               (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"State: {state}",                              (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            else:
                cv2.putText(frame, "No Face Detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("MindSync - Head Pose", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_head_pose()