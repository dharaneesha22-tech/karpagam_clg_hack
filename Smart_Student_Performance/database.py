"""
database.py - SQLite database initialization and all CRUD operations.
"""

import sqlite3
import hashlib
import os
from pathlib import Path
from datetime import datetime

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "database"
DB_PATH = DB_DIR / "students.db"


def get_connection():
    """Return a new SQLite connection with row_factory set."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ── Table Creation ────────────────────────────────────────────────────────────

def init_db():
    """Create all tables if they don't exist."""
    conn = get_connection()
    try:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT    NOT NULL UNIQUE,
                password_hash TEXT    NOT NULL,
                role          TEXT    NOT NULL CHECK(role IN ('student','faculty','admin')),
                created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                roll_number           TEXT    NOT NULL UNIQUE,
                name                  TEXT    NOT NULL,
                department            TEXT    NOT NULL,
                year                  INTEGER NOT NULL,
                semester              INTEGER NOT NULL,
                attendance            REAL    NOT NULL DEFAULT 0,
                internal_mark         REAL    NOT NULL DEFAULT 0,
                assignment_mark       REAL    NOT NULL DEFAULT 0,
                previous_mark         REAL    NOT NULL DEFAULT 0,
                study_hours           REAL    NOT NULL DEFAULT 0,
                assignment_completion REAL    NOT NULL DEFAULT 0,
                final_mark            REAL    NOT NULL DEFAULT 0
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS faculty (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                faculty_id  TEXT    NOT NULL UNIQUE,
                name        TEXT    NOT NULL,
                department  TEXT    NOT NULL,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                roll_number     TEXT    NOT NULL,
                predicted_mark  REAL    NOT NULL,
                grade           TEXT    NOT NULL,
                performance     TEXT    NOT NULL,
                model_name      TEXT    NOT NULL,
                prediction_date TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)

        conn.commit()
    finally:
        conn.close()


# ── Password Hashing ──────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    """Hash a password using PBKDF2-HMAC-SHA256."""
    salt = b"smart_student_salt_v1"
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 260000)
    return dk.hex()


def verify_password(password: str, stored_hash: str) -> bool:
    """Return True if the password matches the stored hash."""
    return hash_password(password) == stored_hash


# ── User / Auth ───────────────────────────────────────────────────────────────

def create_user(username: str, password: str, role: str) -> bool:
    """Insert a new user. Returns False if username already exists."""
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, hash_password(password), role),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_user(username: str):
    """Return user row by username or None."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def user_exists(username: str) -> bool:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def update_user_password(username: str, new_password: str) -> bool:
    conn = get_connection()
    try:
        cur = conn.execute(
            "UPDATE users SET password_hash = ? WHERE username = ?",
            (hash_password(new_password), username),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_user(username: str) -> bool:
    conn = get_connection()
    try:
        cur = conn.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# ── Students ──────────────────────────────────────────────────────────────────

def add_student(roll_number, name, department, year, semester,
                attendance, internal_mark, assignment_mark,
                previous_mark, study_hours, assignment_completion,
                final_mark=0) -> bool:
    """Insert student. Returns False on duplicate roll_number."""
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO students
               (roll_number, name, department, year, semester,
                attendance, internal_mark, assignment_mark,
                previous_mark, study_hours, assignment_completion, final_mark)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (roll_number, name, department, int(year), int(semester),
             float(attendance), float(internal_mark), float(assignment_mark),
             float(previous_mark), float(study_hours),
             float(assignment_completion), float(final_mark)),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_all_students():
    """Return list of all student dicts."""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM students ORDER BY roll_number").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_student_by_roll(roll_number: str):
    """Return student dict or None."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM students WHERE roll_number = ?", (roll_number,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_student(roll_number, name, department, year, semester,
                   attendance, internal_mark, assignment_mark,
                   previous_mark, study_hours, assignment_completion,
                   final_mark) -> bool:
    conn = get_connection()
    try:
        cur = conn.execute(
            """UPDATE students SET
               name=?, department=?, year=?, semester=?,
               attendance=?, internal_mark=?, assignment_mark=?,
               previous_mark=?, study_hours=?, assignment_completion=?,
               final_mark=?
               WHERE roll_number=?""",
            (name, department, int(year), int(semester),
             float(attendance), float(internal_mark), float(assignment_mark),
             float(previous_mark), float(study_hours),
             float(assignment_completion), float(final_mark), roll_number),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_student(roll_number: str) -> bool:
    conn = get_connection()
    try:
        cur = conn.execute(
            "DELETE FROM students WHERE roll_number = ?", (roll_number,)
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def get_students_count() -> int:
    conn = get_connection()
    try:
        row = conn.execute("SELECT COUNT(*) as cnt FROM students").fetchone()
        return row["cnt"] if row else 0
    finally:
        conn.close()


def student_exists(roll_number: str) -> bool:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id FROM students WHERE roll_number = ?", (roll_number,)
        ).fetchone()
        return row is not None
    finally:
        conn.close()


# ── Faculty ───────────────────────────────────────────────────────────────────

def add_faculty(faculty_id: str, name: str, department: str, password: str) -> bool:
    """Insert faculty record and create user account. Returns False on duplicate."""
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO faculty (faculty_id, name, department) VALUES (?,?,?)",
            (faculty_id, name, department),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_all_faculty():
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM faculty ORDER BY faculty_id").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_faculty_by_id(faculty_id: str):
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM faculty WHERE faculty_id = ?", (faculty_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def delete_faculty(faculty_id: str) -> bool:
    conn = get_connection()
    try:
        cur = conn.execute(
            "DELETE FROM faculty WHERE faculty_id = ?", (faculty_id,)
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def get_faculty_count() -> int:
    conn = get_connection()
    try:
        row = conn.execute("SELECT COUNT(*) as cnt FROM faculty").fetchone()
        return row["cnt"] if row else 0
    finally:
        conn.close()


def faculty_exists(faculty_id: str) -> bool:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id FROM faculty WHERE faculty_id = ?", (faculty_id,)
        ).fetchone()
        return row is not None
    finally:
        conn.close()


# ── Predictions ───────────────────────────────────────────────────────────────

def save_prediction(roll_number: str, predicted_mark: float,
                    grade: str, performance: str, model_name: str) -> bool:
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO predictions
               (roll_number, predicted_mark, grade, performance, model_name, prediction_date)
               VALUES (?,?,?,?,?,?)""",
            (roll_number, float(predicted_mark), grade, performance,
             model_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def get_predictions_by_roll(roll_number: str):
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE roll_number = ? ORDER BY prediction_date DESC",
            (roll_number,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_all_predictions():
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY prediction_date DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_predictions_count() -> int:
    conn = get_connection()
    try:
        row = conn.execute("SELECT COUNT(*) as cnt FROM predictions").fetchone()
        return row["cnt"] if row else 0
    finally:
        conn.close()


def get_latest_prediction(roll_number: str):
    conn = get_connection()
    try:
        row = conn.execute(
            """SELECT * FROM predictions WHERE roll_number = ?
               ORDER BY prediction_date DESC LIMIT 1""",
            (roll_number,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ── Bulk / Import ─────────────────────────────────────────────────────────────

def bulk_add_students(records: list) -> tuple:
    """
    Insert multiple student dicts. Returns (success_count, skip_count).
    Each dict must have all student fields.
    """
    success, skipped = 0, 0
    for rec in records:
        ok = add_student(
            rec["roll_number"], rec["name"], rec["department"],
            rec["year"], rec["semester"], rec["attendance"],
            rec["internal_mark"], rec["assignment_mark"],
            rec["previous_mark"], rec["study_hours"],
            rec["assignment_completion"], rec.get("final_mark", 0),
        )
        if ok:
            success += 1
        else:
            skipped += 1
    return success, skipped
