import hashlib
import hmac
import os
import sqlite3
from datetime import datetime

from .config import DATA_DIR


DB_PATH = DATA_DIR / "dialect_app.db"
PBKDF2_ITERATIONS = 260_000


def _connect():
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _connect() as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS recognition_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                dialect TEXT NOT NULL,
                confidence REAL NOT NULL,
                is_reliable INTEGER NOT NULL,
                duration_seconds REAL NOT NULL,
                processing_seconds REAL NOT NULL,
                classifier TEXT NOT NULL DEFAULT 'cnn',
                asr_text TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """
        )
        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(recognition_history)"
            ).fetchall()
        }
        if "classifier" not in columns:
            connection.execute(
                "ALTER TABLE recognition_history "
                "ADD COLUMN classifier TEXT NOT NULL DEFAULT 'cnn'"
            )


def _hash_password(password, salt):
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return digest.hex()


def create_user(username, password):
    username = username.strip()
    if not 3 <= len(username) <= 24:
        return False, "用户名长度需要为 3 到 24 个字符。"
    if len(password) < 6:
        return False, "密码至少需要 6 个字符。"

    salt = os.urandom(16)
    try:
        with _connect() as connection:
            connection.execute(
                """
                INSERT INTO users (username, password_hash, salt, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    username,
                    _hash_password(password, salt),
                    salt.hex(),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
    except sqlite3.IntegrityError:
        return False, "该用户名已经存在。"
    return True, "注册成功，请登录。"


def authenticate_user(username, password):
    with _connect() as connection:
        user = connection.execute(
            """
            SELECT id, username, password_hash, salt
            FROM users
            WHERE username = ?
            """,
            (username.strip(),),
        ).fetchone()

    if user is None:
        return None

    candidate = _hash_password(password, bytes.fromhex(user["salt"]))
    if not hmac.compare_digest(candidate, user["password_hash"]):
        return None
    return {"id": user["id"], "username": user["username"]}


def add_history(user_id, filename, result, processing_seconds):
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO recognition_history (
                user_id, filename, dialect, confidence, is_reliable,
                duration_seconds, processing_seconds, classifier,
                asr_text, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                filename,
                result["dialect_name"],
                result["confidence"],
                int(result["is_reliable"]),
                result["duration_seconds"],
                processing_seconds,
                result.get("classifier", "cnn"),
                result["asr_text"],
                datetime.now().isoformat(timespec="seconds"),
            ),
        )


def get_history(user_id, limit=100):
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT id, filename, dialect, confidence, is_reliable,
                   duration_seconds, processing_seconds, classifier,
                   asr_text, created_at
            FROM recognition_history
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def delete_history(user_id):
    with _connect() as connection:
        connection.execute(
            "DELETE FROM recognition_history WHERE user_id = ?",
            (user_id,),
        )
