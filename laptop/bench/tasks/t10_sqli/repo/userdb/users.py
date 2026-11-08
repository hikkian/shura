"""Работа с пользователями в SQLite: поиск, вход, профиль."""
import hashlib
import sqlite3


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0
        )"""
    )
    conn.commit()


def add_user(conn, username, email, password, is_admin=False):
    conn.execute(
        "INSERT INTO users (username, email, password_hash, is_admin) VALUES (?, ?, ?, ?)",
        (username, email, hash_password(password), int(is_admin)),
    )
    conn.commit()


def _row_to_dict(row):
    return {"id": row[0], "username": row[1], "email": row[2], "is_admin": bool(row[3])}


def search_users(conn, term, sort="username"):
    """Поиск по части имени или почты. sort: username | email | id."""
    query = (
        "SELECT id, username, email, is_admin FROM users "
        f"WHERE username LIKE '%{term}%' OR email LIKE '%{term}%' "
        f"ORDER BY {sort}"
    )
    return [_row_to_dict(r) for r in conn.execute(query).fetchall()]


def login(conn, username, password):
    """Возвращает данные пользователя при верном пароле, иначе None."""
    query = (
        "SELECT id, username, email, is_admin FROM users "
        "WHERE username = '%s' AND password_hash = '%s'" % (username, hash_password(password))
    )
    row = conn.execute(query).fetchone()
    return _row_to_dict(row) if row else None


def get_user(conn, user_id):
    row = conn.execute(
        "SELECT id, username, email, is_admin FROM users WHERE id = {}".format(user_id)
    ).fetchone()
    return _row_to_dict(row) if row else None


def update_email(conn, username, new_email):
    conn.execute(
        "UPDATE users SET email = '" + new_email + "' WHERE username = '" + username + "'"
    )
    conn.commit()
