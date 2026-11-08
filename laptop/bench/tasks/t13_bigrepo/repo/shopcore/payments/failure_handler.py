"""Единая точка обработки неуспешного платежа."""
from ..models.order import STATUS_PAYMENT_FAILED
from ..repositories.order_repo import order_repo
from ..services import inventory_service, notification_service
from ..tasks.retry_payments import schedule_retry
from ..utils.logging_utils import get_logger
from . import retry_policy

log = get_logger("payments.failure")


def handle_payment_failure(order, payment, error):
    """Статус заказа -> payment_failed, снять резерв, уведомить клиента, при возможности запланировать повтор."""
    log.warning("payment failed for order %s: %s", order.id, error)
    order.status = STATUS_PAYMENT_FAILED
    order_repo.add(order)
    inventory_service.release(order.items)
    notification_service.send_payment_failed(order, str(error))
    if retry_policy.should_retry(error, payment.attempts):
        schedule_retry(order.id, retry_policy.delay_for(payment.attempts))
        return "retry_scheduled"
    return "failed"
