from __future__ import annotations
import json
import sqlite3
import time
from pathlib import Path
import numpy as np
from config import DB_PATH, DATA_DIR, PHOTOS_DIR, MODELS_DIR

def connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def init_db():
    with connect() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS visitors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            external_ref TEXT DEFAULT '',
            photo_path TEXT NOT NULL,
            embedding BLOB,
            consent_confirmed INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            event_type TEXT NOT NULL,
            visitor_id INTEGER,
            visitor_name TEXT,
            track_id INTEGER,
            confidence REAL,
            note TEXT DEFAULT '',
            FOREIGN KEY(visitor_id) REFERENCES visitors(id) ON DELETE SET NULL
        );
        CREATE TABLE IF NOT EXISTS occupancy_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            people_count INTEGER NOT NULL,
            recognized_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """)
        columns = {r["name"] for r in c.execute("PRAGMA table_info(occupancy_samples)").fetchall()}
        if "recognized_count" not in columns:
            c.execute("ALTER TABLE occupancy_samples ADD COLUMN recognized_count INTEGER NOT NULL DEFAULT 0")
    migrate_photo_paths()

def resolve_photo_path(stored_path: str) -> str:
    if not stored_path:
        return stored_path
    p = Path(stored_path)
    if p.exists():
        return str(p)
    local_p = PHOTOS_DIR / p.name
    if local_p.exists():
        return str(local_p)
    return str(p)

def migrate_photo_paths():
    try:
        with connect() as c:
            rows = c.execute("SELECT id, photo_path FROM visitors").fetchall()
            for r in rows:
                old_path = r["photo_path"]
                if old_path:
                    p = Path(old_path)
                    if not p.exists():
                        local_p = PHOTOS_DIR / p.name
                        if local_p.exists():
                            c.execute("UPDATE visitors SET photo_path=? WHERE id=?", (str(local_p), r["id"]))
    except Exception:
        pass

def get_settings(defaults: dict) -> dict:
    result = dict(defaults)
    with connect() as c:
        rows = c.execute("SELECT key, value FROM settings").fetchall()
    for row in rows:
        try:
            result[row["key"]] = json.loads(row["value"])
        except Exception:
            result[row["key"]] = row["value"]
    return result

def save_settings(values: dict):
    with connect() as c:
        c.executemany(
            "INSERT INTO settings(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            [(str(k), json.dumps(v)) for k, v in values.items()]
        )

def add_visitor(name, photo_path, external_ref="", embedding=None, consent_confirmed=False):
    with connect() as c:
        cur = c.execute(
            """INSERT INTO visitors(name,external_ref,photo_path,embedding,consent_confirmed,created_at)
               VALUES(?,?,?,?,?,?)""",
            (name.strip(), external_ref.strip(), str(photo_path),
             embedding.astype(np.float32).tobytes() if embedding is not None else None,
             int(consent_confirmed), time.time())
        )
        return int(cur.lastrowid)

def update_visitor_embedding(visitor_id: int, embedding):
    blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
    with connect() as c:
        c.execute("UPDATE visitors SET embedding=? WHERE id=?", (blob, visitor_id))

def list_visitors():
    with connect() as c:
        rows = c.execute("SELECT * FROM visitors ORDER BY name COLLATE NOCASE").fetchall()
    result = []
    for row in rows:
        item = dict(row)
        raw = item.get("embedding")
        item["embedding_array"] = np.frombuffer(raw, dtype=np.float32).copy() if raw else None
        item["photo_path"] = resolve_photo_path(item.get("photo_path", ""))
        result.append(item)
    return result

def delete_visitor(visitor_id: int):
    with connect() as c:
        row = c.execute("SELECT photo_path FROM visitors WHERE id=?", (visitor_id,)).fetchone()
        c.execute("DELETE FROM visitors WHERE id=?", (visitor_id,))
    if row and row["photo_path"]:
        resolved = resolve_photo_path(row["photo_path"])
        try:
            Path(resolved).unlink(missing_ok=True)
        except OSError:
            pass

def log_event(event_type, visitor_id=None, visitor_name=None, track_id=None, confidence=None, note=""):
    with connect() as c:
        c.execute(
            """INSERT INTO events(timestamp,event_type,visitor_id,visitor_name,track_id,confidence,note)
               VALUES(?,?,?,?,?,?,?)""",
            (time.time(), event_type, visitor_id, visitor_name, track_id, confidence, note)
        )

def log_occupancy(people_count, recognized_count=0):
    with connect() as c:
        c.execute(
            "INSERT INTO occupancy_samples(timestamp,people_count,recognized_count) VALUES(?,?,?)",
            (time.time(), int(people_count), int(recognized_count))
        )

def get_events(limit=500):
    with connect() as c:
        rows = c.execute("SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (int(limit),)).fetchall()
    return [dict(r) for r in rows]

def get_occupancy(limit=600):
    with connect() as c:
        rows = c.execute(
            "SELECT timestamp,people_count,recognized_count FROM occupancy_samples "
            "ORDER BY timestamp DESC LIMIT ?", (int(limit),)
        ).fetchall()
    return [dict(r) for r in reversed(rows)]

def get_occupancy_since(since_timestamp, limit=1000):
    with connect() as c:
        rows = c.execute(
            "SELECT timestamp,people_count,recognized_count FROM occupancy_samples "
            "WHERE timestamp >= ? ORDER BY timestamp ASC LIMIT ?",
            (float(since_timestamp), int(limit))
        ).fetchall()
    return [dict(r) for r in rows]

def get_peak_occupancy():
    with connect() as c:
        row = c.execute("SELECT MAX(people_count) AS peak FROM occupancy_samples").fetchone()
    if row and row["peak"] is not None:
        return int(row["peak"])
    return 0

def event_counts_since(timestamp):
    with connect() as c:
        row = c.execute("""
            SELECT
              SUM(CASE WHEN event_type='track_started' THEN 1 ELSE 0 END) AS tracks,
              SUM(CASE WHEN event_type='recognized' THEN 1 ELSE 0 END) AS recognitions
            FROM events WHERE timestamp >= ?
        """, (timestamp,)).fetchone()
    return {"tracks": int(row["tracks"] or 0), "recognitions": int(row["recognitions"] or 0)}

def clear_analytics():
    with connect() as c:
        c.execute("DELETE FROM events")
        c.execute("DELETE FROM occupancy_samples")

