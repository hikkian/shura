from decimal import Decimal

from ..config.settings import FREE_SHIPPING_THRESHOLD

BASE = {"KZ": Decimal("1500"), "RU": Decimal("3500"), "DE": Decimal("9000")}


def calculate_shipping_cost(order_total, weight_kg, country):
    if order_total >= FREE_SHIPPING_THRESHOLD:
        return Decimal(0)
    return BASE.get(country, Decimal("12000")) + Decimal(str(weight_kg)) * Decimal("200")
