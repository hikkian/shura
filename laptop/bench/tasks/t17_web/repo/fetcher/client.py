import httpx


def fetch_json(url, timeout=10.0):
    """Скачивает JSON по адресу и возвращает разобранный объект."""
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.json()
