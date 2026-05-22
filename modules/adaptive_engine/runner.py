import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from modules.cognitive_detection.state_classifier import (
    get_eye_points, calculate_ear, get_head_pose, classify_state,
    LEFT_EYE, RIGHT_EYE, PITCH_OFFSET, YAW_OFFSET,
    EAR_THRESHOLD, FATIGUE_FRAMES, BLINK_FRAMES, STATE_COLORS
)
from modules.adaptive_engine.engine import AdaptiveEngine

BaseOptions       = mp.tasks.BaseOptions
FaceLandmarker    = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def draw_recommendation(frame, recommendation):
    """Draw the recommendation banner at the bottom of the frame."""
    if recommendation is None:
        return

    fh, fw, _ = frame.shape
    state   = recommendation["state"]
    message = recommendation["message"]
    color   = STATE_COLORS[state]

    # Draw dark background banner
    cv2.rectangle(frame, (0, fh - 60), (fw, fh), (0, 0, 0), -1)

    # Draw message
    cv2.putText(frame, message, (10, fh - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def run():
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path='models/face_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1)

    cap    = cv2.VideoCapture(0)
    engine = AdaptiveEngine()

    closed_frames = 0
    blink_times   = []
    prev_ear_low  = False

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

                if avg_ear < EAR_THRESHOLD:
                    closed_frames += 1
                    prev_ear_low   = True
                else:
                    if prev_ear_low:
                        if closed_frames <= BLINK_FRAMES * 3:
                            blink_times.append(time.time())
                    closed_frames = 0
                    prev_ear_low  = False

                now          = time.time()
                blink_times  = [t for t in blink_times if now - t <= 5]
                recent_blinks = len(blink_times)

                # Head pose
                pitch, yaw = get_head_pose(lm, fw, fh)
                pitch_adj  = pitch - PITCH_OFFSET
                yaw_adj    = yaw   - YAW_OFFSET

                # Classify
                state = classify_state(
                    avg_ear, pitch_adj, yaw_adj,
                    closed_frames, recent_blinks)
                color = STATE_COLORS[state]

                # Adaptive engine
                recommendation = engine.update(state)

                # Display metrics
                cv2.putText(frame, f"EAR: {round(avg_ear,3)}",        (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                cv2.putText(frame, f"Blinks (5s): {recent_blinks}",   (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                cv2.putText(frame, f"Pitch adj: {round(pitch_adj,1)}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                cv2.putText(frame, f"Yaw adj:   {round(yaw_adj,1)}",   (10,105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                cv2.putText(frame, f"STATE: {state}",                  (10,145),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

                # Show last recommendation persistently
                draw_recommendation(frame,
                    recommendation or engine.get_last_recommendation())

                # Eye dots
                for p in left_pts + right_pts:
                    cv2.circle(frame, p, 2, color, -1)

            else:
                cv2.putText(frame, "No Face Detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            cv2.imshow("MindSync - Adaptive Learning", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Print session log on exit
    print("\n--- Session Action Log ---")
    for entry in engine.get_log():
        t = time.strftime('%H:%M:%S', time.localtime(entry['timestamp']))
        print(f"[{t}] {entry['state']:12} → {entry['action']}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()