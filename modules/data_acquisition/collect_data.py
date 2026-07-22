import cv2
import mediapipe as mp
import numpy as np
import csv
import time
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.cognitive_detection.state_classifier import (
    get_eye_points, calculate_ear, get_head_pose,
    LEFT_EYE, RIGHT_EYE, PITCH_OFFSET, YAW_OFFSET,
    EAR_THRESHOLD, BLINK_FRAMES
)

BaseOptions       = mp.tasks.BaseOptions
FaceLandmarker    = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Output CSV path
OUTPUT_PATH = "data/raw/session_data.csv"
FIELDNAMES  = [
    "timestamp", "ear", "pitch_adj", "yaw_adj",
    "blink_rate", "closed_frames", "label"
]

# Key → label mapping
KEY_LABELS = {
    ord('f'): "Focused",
    ord('d'): "Distracted",
    ord('t'): "Fatigued",
    ord('c'): "Confused",
}

LABEL_COLORS = {
    "Focused":    (0,   255, 0  ),
    "Distracted": (0,   165, 255),
    "Fatigued":   (0,   0,   255),
    "Confused":   (255, 0,   255),
    None:         (128, 128, 128),
}

def collect():
    # Create CSV with headers if it doesn't exist
    file_exists = os.path.exists(OUTPUT_PATH)
    csvfile = open(OUTPUT_PATH, "a", newline="")
    writer  = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path="models/face_landmarker.task"),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1)

    cap = cv2.VideoCapture(0)

    closed_frames = 0
    blink_times   = []
    blink_count   = 0
    prev_ear_low  = False
    current_label = None
    frame_count   = 0
    saved_count   = 0

    print("Data collection started!")
    print("Press F=Focused, D=Distracted, T=Fatigued, C=Confused, Q=Quit")

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            fh, fw, _ = frame.shape
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=rgb)
            results  = landmarker.detect(mp_image)

            if results.face_landmarks:
                lm = results.face_landmarks[0]

                # Extract features
                left_pts  = get_eye_points(lm, LEFT_EYE,  fw, fh)
                right_pts = get_eye_points(lm, RIGHT_EYE, fw, fh)
                avg_ear   = (calculate_ear(left_pts) +
                             calculate_ear(right_pts)) / 2.0

                if avg_ear < EAR_THRESHOLD:
                    closed_frames += 1
                    prev_ear_low   = True
                else:
                    if prev_ear_low:
                        if closed_frames <= BLINK_FRAMES * 3:
                            blink_count += 1
                            blink_times.append(time.time())
                    closed_frames = 0
                    prev_ear_low  = False

                now         = time.time()
                blink_times = [t for t in blink_times
                               if now - t <= 5]
                blink_rate  = len(blink_times)

                pitch, yaw = get_head_pose(lm, fw, fh)
                pitch_adj  = pitch - PITCH_OFFSET
                yaw_adj    = yaw   - YAW_OFFSET

                # Save row if label is set
                if current_label is not None:
                    frame_count += 1
                    # Save every 5th frame to avoid redundant data
                    if frame_count % 5 == 0:
                        writer.writerow({
                            "timestamp":    round(now, 3),
                            "ear":          round(avg_ear, 4),
                            "pitch_adj":    round(pitch_adj, 2),
                            "yaw_adj":      round(yaw_adj, 2),
                            "blink_rate":   blink_rate,
                            "closed_frames":closed_frames,
                            "label":        current_label
                        })
                        csvfile.flush()
                        saved_count += 1

                # Display
                color = LABEL_COLORS[current_label]
                cv2.putText(frame,
                    f"EAR: {round(avg_ear,3)}",     (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                cv2.putText(frame,
                    f"Pitch adj: {round(pitch_adj,1)}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                cv2.putText(frame,
                    f"Yaw adj: {round(yaw_adj,1)}",  (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                cv2.putText(frame,
                    f"Blink rate: {blink_rate}",     (10,105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                cv2.putText(frame,
                    f"Label: {current_label or 'None — press key'}",
                    (10, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame,
                    f"Saved: {saved_count} rows",    (10,175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

                # Border color = current label
                cv2.rectangle(frame, (0,0),
                    (fw-1, fh-1), color, 3)

            else:
                cv2.putText(frame, "No Face Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # Instructions
            cv2.putText(frame,
                "F=Focused D=Distracted T=Fatigued C=Confused Q=Quit",
                (10, fh - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

            cv2.imshow("MindSync - Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in KEY_LABELS:
                current_label = KEY_LABELS[key]
                print(f"Label set to: {current_label}")

    cap.release()
    cv2.destroyAllWindows()
    csvfile.close()
    print(f"\nDone! {saved_count} rows saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    collect()