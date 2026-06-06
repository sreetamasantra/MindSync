import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.cognitive_detection.state_classifier import (
    get_eye_points, calculate_ear, get_head_pose, classify_state,
    LEFT_EYE, RIGHT_EYE, PITCH_OFFSET, YAW_OFFSET,
    EAR_THRESHOLD, FATIGUE_FRAMES, BLINK_FRAMES
)
from modules.adaptive_engine.engine import AdaptiveEngine
from modules.analytics.session_state import session

BaseOptions       = mp.tasks.BaseOptions
FaceLandmarker    = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

_stop_event = threading.Event()

def detection_loop():
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path='models/face_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1)

    cap    = cv2.VideoCapture(0)
    engine = AdaptiveEngine()

    closed_frames = 0
    blink_times   = []
    blink_count   = 0
    prev_ear_low  = False

    with FaceLandmarker.create_from_options(options) as landmarker:
        while not _stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                continue

            fh, fw, _ = frame.shape
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=rgb)
            results  = landmarker.detect(mp_image)

            if results.face_landmarks:
                lm = results.face_landmarks[0]

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

                now          = time.time()
                blink_times  = [t for t in blink_times
                                if now - t <= 5]
                recent_blinks = len(blink_times)

                pitch, yaw = get_head_pose(lm, fw, fh)
                pitch_adj  = pitch - PITCH_OFFSET
                yaw_adj    = yaw   - YAW_OFFSET

                state = classify_state(
                    avg_ear, pitch_adj, yaw_adj,
                    closed_frames, recent_blinks)

                session.update_detection(
                    state, avg_ear, pitch_adj,
                    yaw_adj, blink_count, recent_blinks)

                recommendation = engine.update(state)
                session.update_recommendation(recommendation)

            else:
                session.update_detection(
                    "No Face", 0, 0, 0, 0, 0)

    cap.release()

def start_detection_thread():
    _stop_event.clear()
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()
    return t

def stop_detection_thread():
    _stop_event.set()