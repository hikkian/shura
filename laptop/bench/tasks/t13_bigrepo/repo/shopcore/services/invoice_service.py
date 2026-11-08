from ..models.invoice import Invoice
from ..utils.ids import next_id


def build_invoice(order):
    """Счёт берёт уже посчитанную при оформлении цену из order.price."""
    lines = [f"{i.name} x{i.quantity}" for i in order.items]
    return Invoice(next_id("invoice"), order.id, order.price["total"], lines)
