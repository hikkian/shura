class PaymentError(Exception):
    retryable = False


class PaymentDeclined(PaymentError):
    """Карта отклонена банком — повторять бессмысленно."""


class GatewayTimeout(PaymentError):
    """Шлюз не ответил — можно повторить позже."""
    retryable = True
