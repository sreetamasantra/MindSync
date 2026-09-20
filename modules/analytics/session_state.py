import time
import threading
from modules.analytics.database import (
    init_db, start_session, log_event, end_session
)

class SessionState:
    """Thread-safe shared state between detection and Flask."""

    def __init__(self):
        self._lock           = threading.Lock()
        self._state          = "Initializing"
        self._ear            = 0.0
        self._pitch_adj      = 0.0
        self._yaw_adj        = 0.0
        self._blink_count    = 0
        self._recent_blinks  = 0
        self._last_updated   = time.time()
        self._action_log     = []
        self._recommendation = None
        self._confidence     = None

        # Database
        init_db()
        self._session_id = start_session()
        print(f"Session started — ID: {self._session_id}")

    def update_detection(self, state, ear, pitch_adj,
                         yaw_adj, blink_count, recent_blinks):
        with self._lock:
            self._state         = state
            self._ear           = round(ear, 3)
            self._pitch_adj     = round(pitch_adj, 1)
            self._yaw_adj       = round(yaw_adj, 1)
            self._blink_count   = blink_count
            self._recent_blinks = recent_blinks
            self._last_updated  = time.time()

    def update_recommendation(self, recommendation):
        with self._lock:
            if recommendation:
                self._recommendation = recommendation
                self._action_log.append(recommendation)

                # Persist to database
                log_event(
                    session_id  = self._session_id,
                    state       = recommendation["state"],
                    action      = recommendation["action"],
                    message     = recommendation["message"],
                    ear         = self._ear,
                    pitch_adj   = self._pitch_adj,
                    yaw_adj     = self._yaw_adj,
                    confidence  = self._confidence
                )

    def get_current(self):
        with self._lock:
            return {
                "state":          self._state,
                "ear":            self._ear,
                "pitch_adj":      self._pitch_adj,
                "yaw_adj":        self._yaw_adj,
                "blink_count":    self._blink_count,
                "recent_blinks":  self._recent_blinks,
                "last_updated":   self._last_updated,
                "recommendation": self._recommendation,
                "confidence":     self._confidence,
                "session_id":     self._session_id
            }

    def get_log(self):
        with self._lock:
            return list(self._action_log)

    def get_last_recommendation(self):
        with self._lock:
            return self._recommendation

    def reset(self):
        with self._lock:
            end_session(self._session_id)
            self._action_log     = []
            self._recommendation = None
            self._blink_count    = 0
            self._state          = "Initializing"
            self._session_id     = start_session()
            print(f"Session reset — new ID: {self._session_id}")

# Global singleton
session = SessionState()