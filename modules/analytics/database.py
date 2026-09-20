import sqlite3
import time
import os

DB_PATH = "data/mindsync.db"

def get_connection():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at  REAL NOT NULL,
            ended_at    REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  INTEGER NOT NULL,
            timestamp   REAL NOT NULL,
            state       TEXT NOT NULL,
            action      TEXT NOT NULL,
            message     TEXT NOT NULL,
            ear         REAL,
            pitch_adj   REAL,
            yaw_adj     REAL,
            confidence  REAL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

def start_session():
    """Create a new session and return its ID."""
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO sessions (started_at) VALUES (?)",
        (time.time(),)
    )
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id

def end_session(session_id):
    """Mark a session as ended."""
    conn = get_connection()
    conn.execute(
        "UPDATE sessions SET ended_at = ? WHERE id = ?",
        (time.time(), session_id)
    )
    conn.commit()
    conn.close()

def log_event(session_id, state, action, message,
              ear=None, pitch_adj=None,
              yaw_adj=None, confidence=None):
    """Log a cognitive state event."""
    conn = get_connection()
    conn.execute("""
        INSERT INTO events
        (session_id, timestamp, state, action,
         message, ear, pitch_adj, yaw_adj, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (session_id, time.time(), state, action,
          message, ear, pitch_adj, yaw_adj, confidence))
    conn.commit()
    conn.close()

def get_session_events(session_id):
    """Get all events for a session."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM events WHERE session_id = ? ORDER BY timestamp",
        (session_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_all_sessions():
    """Get all sessions with event counts."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT s.id, s.started_at, s.ended_at,
               COUNT(e.id) as event_count
        FROM sessions s
        LEFT JOIN events e ON e.session_id = s.id
        GROUP BY s.id
        ORDER BY s.started_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_recent_events(limit=50):
    """Get most recent events across all sessions."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT e.*, s.started_at as session_start
        FROM events e
        JOIN sessions s ON s.id = e.session_id
        ORDER BY e.timestamp DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]