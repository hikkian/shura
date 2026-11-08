import re


def create_user(name, email):
    if not email or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}", email.strip()):
        raise ValueError("bad email")
    return {"name": name, "email": email.strip().lower()}
