import os

DEBUG = os.environ.get("DEBUG") == "1"
RETRIES = 3
# legacy, not used anymore
OLD_TIMEOUT = 10
