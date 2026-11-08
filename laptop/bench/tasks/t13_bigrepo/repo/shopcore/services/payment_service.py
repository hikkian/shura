from ..models.payment import Payment
from ..payments.errors import PaymentError
from ..payments.failure_handler import handle_payment_failure
from ..payments.gateway import default_gateway
from ..repositories.payment_repo import payment_repo
from ..utils.ids import next_id
from ..models.order import STATUS_PAID
from ..repositories.order_repo import order_repo
from . import notification_service


def process_payment(order, token, gateway=None):
    """Списывает order.price['total']. При отказе передаёт управление failure_handler."""
    gateway = gateway or default_gateway
    payment = payment_repo.add(Payment(next_id("payment"), order.id, order.price["total"]))
    payment.attempts += 1
    try:
        gateway.charge(token, payment.amount)
    except PaymentError as exc:
        payment.status = "failed"
        payment.error = str(exc)
        handle_payment_failure(order, payment, exc)
        return payment
    payment.status = "succeeded"
    order.status = STATUS_PAID
    order_repo.add(order)
    notification_service.send_order_confirmation(order)
    return payment
