def truncate(text, limit=80):
    return text if len(text) <= limit else text[: limit - 1] + "…"
