from flask import Flask, jsonify
from flask_cors import CORS
import threading

from modules.analytics.session_state import session
from modules.analytics.detection_thread import (
    start_detection_thread, stop_detection_thread)

app = Flask(__name__)
CORS(app)  # Allow React frontend to call this API

# Routes 

@app.route('/api/state', methods=['GET'])
def get_state():
    """Returns current cognitive state and metrics."""
    return jsonify(session.get_current())

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