import urllib.request

from app.config import RETRIES

DEFAULT_TIMEOUT = 25  # seconds


def get(url, timeout=None):
    t = DEFAULT_TIMEOUT if timeout is None else timeout
    for attempt in range(RETRIES):
        try:
            return urllib.request.urlopen(url, timeout=t).read()
        except OSError:
            if attempt == RETRIES - 1:
                raise
