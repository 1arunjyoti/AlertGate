import os
import sqlite3
import threading
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class EventStore:
    """SQLite-backed store for detection events.

    Schema:
      events(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,         -- ISO 8601 string
        class_name TEXT NOT NULL,
        confidence REAL NOT NULL,
        frame_number INTEGER,
        zone TEXT,
        snapshot_path TEXT
      )
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        # Ensure parent directory exists
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA temp_store=MEMORY;")
        self._init_schema()

    def _init_schema(self):
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    frame_number INTEGER,
                    zone TEXT,
                    snapshot_path TEXT
                );
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC);"
            )
            # Prevent duplicates for the same logical event; if this fails due to existing
            try:
                self._conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_events_dedupe ON events(timestamp, class_name, frame_number, zone);"
                )
            except sqlite3.IntegrityError:
                self._dedupe_existing_rows()
                # Retry creating the unique index after pruning duplicates
                self._conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_events_dedupe ON events(timestamp, class_name, frame_number, zone);"
                )
            self._conn.commit()

    # Remove duplicates keeping the smallest id per logical event key
    def _dedupe_existing_rows(self):
        """Remove duplicate rows keeping the smallest id per logical event key."""
        # Use a DELETE with a subselect to keep MIN(id) for each key group
        self._conn.execute(
            """
            DELETE FROM events
            WHERE id NOT IN (
                SELECT MIN(id) FROM events
                GROUP BY timestamp, class_name, frame_number, zone
            );
            """
        )
        self._conn.commit()

    # Insert a new event
    def add_event(self, event: Dict) -> int:
        """Insert a new event. Returns inserted row id.
        Expected keys in event dict: class_name, confidence, timestamp (ISO str), frame_number, zone, snapshot_path(optional)
        """
        ts = event.get("timestamp")
        if not ts:
            ts = datetime.utcnow().isoformat()
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT OR IGNORE INTO events(timestamp, class_name, confidence, frame_number, zone, snapshot_path)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    event.get("class_name", "unknown"),
                    float(event.get("confidence", 0.0)),
                    int(event.get("frame_number", 0)),
                    event.get("zone"),
                    event.get("snapshot_path"),
                ),
            )
            self._conn.commit()
            # If ignored, rowcount may be 0
            if cur.rowcount == 0:
                return -1
            return int(cur.lastrowid)

    # Retrieve recent events
    def get_recent_events(self, limit: int = 50) -> List[Dict]:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT id, timestamp, class_name, confidence, frame_number, zone, snapshot_path
                FROM events
                ORDER BY datetime(timestamp) DESC
                LIMIT ?
                """,
                (int(limit),),
            )
            rows = cur.fetchall()
        events = []
        for r in rows:
            events.append(
                {
                    "id": r[0],
                    "timestamp": r[1],
                    "class_name": r[2],
                    "confidence": r[3],
                    "frame_number": r[4],
                    "zone": r[5],
                    "snapshot_path": r[6],
                }
            )
        return events

    # Prune events older than given days
    def prune_older_than(self, days: int = 30) -> int:
        """Delete events older than given days. Returns number of rows deleted."""
        cutoff = (datetime.utcnow() - timedelta(days=int(days))).isoformat()
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM events WHERE datetime(timestamp) < datetime(?)",
                (cutoff,),
            )
            self._conn.commit()
            return cur.rowcount or 0

    # Close the connection
    def close(self):
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass
