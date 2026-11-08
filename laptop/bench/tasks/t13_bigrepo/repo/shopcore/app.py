from .api.router import dispatch
from .config.logging_config import setup_logging


def create_app():
    setup_logging()
    return dispatch
