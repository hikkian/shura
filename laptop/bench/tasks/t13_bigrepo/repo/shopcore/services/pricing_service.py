from dataclasses import asdict, dataclass
from decimal import Decimal

from ..utils.money import quantize
from . import discount_service, tax_service


@dataclass
class PriceBreakdown:
    subtotal: Decimal
    discount: Decimal
    discount_percent: int
    tax: Decimal
    total: Decimal

    def as_dict(self):
        return asdict(self)


def calculate_subtotal(items):
    return quantize(sum((i.unit_price * i.quantity for i in items), Decimal(0)))


def calculate_final_price(items, customer, coupon=None, country=None, is_first_order=False):
    """Итоговая цена заказа: подытог -> скидки -> налог на сумму после скидки."""
    subtotal = calculate_subtotal(items)
    count = sum(i.quantity for i in items)
    disc = discount_service.apply_discounts(subtotal, count, customer, coupon, is_first_order)
    after_discount = subtotal - disc.amount
    tax = tax_service.calculate_tax(after_discount, country or customer.country, items)
    total = quantize(after_discount + tax)
    return PriceBreakdown(subtotal, disc.amount, disc.percent, tax, total)
