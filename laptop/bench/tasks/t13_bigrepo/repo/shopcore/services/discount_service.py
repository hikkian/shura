from dataclasses import dataclass, field
from decimal import Decimal

from ..config.discount_rules import DISCOUNT_RULES, MAX_TOTAL_DISCOUNT_PERCENT
from ..utils.money import percent_of


@dataclass
class DiscountResult:
    percent: int
    amount: Decimal
    applied: list = field(default_factory=list)


def _rule_matches(rule, items_count, customer, is_first_order):
    kind = rule["type"]
    if kind == "volume":
        return items_count >= rule["min_items"]
    if kind == "loyalty":
        return customer.tier == rule["tier"]
    if kind == "first_order":
        return is_first_order
    return False


def apply_discounts(subtotal, items_count, customer, coupon=None, is_first_order=False):
    """Суммирует проценты подходящих правил и купона, ограничивает потолком из конфига."""
    percent = 0
    applied = []
    for rule in DISCOUNT_RULES:
        if _rule_matches(rule, items_count, customer, is_first_order):
            percent += rule["percent"]
            applied.append(rule["name"])
    if coupon is not None and coupon.active:
        percent += coupon.percent
        applied.append("coupon:" + coupon.code)
    percent = min(percent, MAX_TOTAL_DISCOUNT_PERCENT)
    return DiscountResult(percent, percent_of(subtotal, percent), applied)
