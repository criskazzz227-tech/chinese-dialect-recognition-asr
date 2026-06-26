import sqlite3

from src import database


def use_temporary_database(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    database.init_db()
    return db_path


def test_password_is_hashed_and_authentication_works(tmp_path, monkeypatch):
    db_path = use_temporary_database(tmp_path, monkeypatch)

    created, _ = database.create_user("alice", "secret123")
    assert created is True
    assert database.authenticate_user("alice", "secret123") is not None
    assert database.authenticate_user("alice", "wrong") is None

    with sqlite3.connect(db_path) as connection:
        stored = connection.execute(
            "SELECT password_hash FROM users WHERE username = 'alice'"
        ).fetchone()[0]
    assert stored != "secret123"


def test_history_is_scoped_to_each_user(tmp_path, monkeypatch):
    use_temporary_database(tmp_path, monkeypatch)
    database.create_user("alice", "secret123")
    database.create_user("bob", "secret123")
    alice = database.authenticate_user("alice", "secret123")
    bob = database.authenticate_user("bob", "secret123")

    result = {
        "dialect_name": "上海话",
        "confidence": 0.8,
        "is_reliable": True,
        "duration_seconds": 3.2,
        "classifier": "whisper",
        "asr_text": "",
    }
    database.add_history(alice["id"], "sample.wav", result, 0.5)

    assert len(database.get_history(alice["id"])) == 1
    assert database.get_history(bob["id"]) == []
