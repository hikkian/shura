OUTBOX = []


def _send(kind, customer_id, payload):
    OUTBOX.append({"kind": kind, "customer_id": customer_id, **payload})


def send_order_confirmation(order):
    _send("order_confirmation", order.customer_id, {"order_id": order.id})


def send_payment_failed(order, reason):
    _send("payment_failed", order.customer_id, {"order_id": order.id, "reason": reason})


def send_shipment_update(order, status):
    _send("shipment", order.customer_id, {"order_id": order.id, "status": status})
