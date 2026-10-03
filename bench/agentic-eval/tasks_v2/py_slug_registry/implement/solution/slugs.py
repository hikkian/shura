import re
import unicodedata

def slugify(text):
    if not isinstance(text, str): raise TypeError("text must be str")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    value = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return value or "item"

def unique_slug(text, used):
    base = slugify(text)
    if base not in used: return base
    n = 2
    while f"{base}-{n}" in used: n += 1
    return f"{base}-{n}"
