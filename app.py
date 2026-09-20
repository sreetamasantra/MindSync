from flask import Flask, jsonify
from flask_cors import CORS
import threading

from modules.analytics.session_state import session
from modules.analytics.detection_thread import (
    start_detection_thread, stop_detection_thread)
from modules.analytics.database import get_all_sessions, get_session_events, get_recent_events

app = Flask(__name__)
CORS(app)  # Allow React frontend to call this API

# Routes 

@app.route('/api/state', methods=['GET']) 
def get_state():
    """Returns current cognitive state and metrics."""
    data = session.get_current()
    data["confidence"] = getattr(session, '_confidence', None)
    return jsonify(data)

@app.route('/api/session/log', methods=['GET'])
def get_log():
    """Returns full session action log."""
    log = session.get_log()
    return jsonify({
        "count": len(log),
        "log":   log
    })

@app.route('/api/session/reset', methods=['POST'])
def reset_session():
    """Resets the session."""
    session.reset()
    return jsonify({"status": "reset successful"})

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Returns all past sessions."""
    sessions = get_all_sessions()
    return jsonify({"sessions": sessions})

@app.route('/api/sessions/<int:session_id>/events', methods=['GET'])
def get_events(session_id):
    """Returns all events for a specific session."""
    events = get_session_events(session_id)
    return jsonify({
        "session_id": session_id,
        "count": len(events),
        "events": events
    })

@app.route('/api/events/recent', methods=['GET'])
def recent_events():
    """Returns 50 most recent events across all sessions."""
    events = get_recent_events(50)
    return jsonify({"events": events})

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "running"})

# Main 

if __name__ == '__main__':
    print("Starting detection thread...")
    start_detection_thread()
    print("Starting Flask server on http://localhost:5000")
    app.run(debug=False, port=5000)