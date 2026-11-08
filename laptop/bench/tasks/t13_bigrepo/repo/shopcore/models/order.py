from dataclasses import dataclass, field
from decimal import Decimal

STATUS_NEW = "new"
STATUS_PAID = "paid"
STATUS_PAYMENT_FAILED = "payment_failed"
STATUS_CANCELLED = "cancelled"


@dataclass
class Order:
    id: int
    customer_id: int
    items: list = field(default_factory=list)
    status: str = STATUS_NEW
    price: dict = field(default_factory=dict)  # разбивка, посчитанная при оформлении
    country: str = "KZ"
    coupon_code: str = ""

    def items_count(self):
        return sum(i.quantity for i in self.items)

    def items_total(self):
        """Сумма позиций без скидок и налога (только для отображения)."""
        return sum((i.unit_price * i.quantity for i in self.items), Decimal(0))
