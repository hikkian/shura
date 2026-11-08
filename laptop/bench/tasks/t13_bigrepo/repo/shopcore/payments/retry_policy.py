from ..config.settings import PAYMENT_RETRY_DELAY_SECONDS, PAYMENT_RETRY_LIMIT


def should_retry(error, attempts):
    return bool(getattr(error, "retryable", False)) and attempts < PAYMENT_RETRY_LIMIT


def delay_for(attempts):
    return PAYMENT_RETRY_DELAY_SECONDS * attempts
