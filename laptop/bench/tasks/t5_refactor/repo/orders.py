import re


def create_order(email, amount):
    if not email or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}", email.strip()):
        raise ValueError("bad email")
    if amount <= 0:
        raise ValueError("bad amount")
    return {"email": email.strip().lower(), "amount": amount}
