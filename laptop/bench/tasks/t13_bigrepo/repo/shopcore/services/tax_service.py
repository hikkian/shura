from decimal import Decimal

from ..config.settings import DEFAULT_COUNTRY
from ..config.tax_rates import DEFAULT_TAX_RATE, TAX_EXEMPT_CATEGORIES, TAX_RATES
from ..utils.money import percent_of


def tax_rate_for(country):
    return TAX_RATES.get(country or DEFAULT_COUNTRY, DEFAULT_TAX_RATE)


def calculate_tax(taxable_amount, country, items=()):
    """Налог начисляется на сумму после скидки; освобождённые категории не облагаются."""
    exempt = sum((i.unit_price * i.quantity for i in items if i.category in TAX_EXEMPT_CATEGORIES), Decimal(0))
    total = sum((i.unit_price * i.quantity for i in items), Decimal(0))
    share = Decimal(1) if total == 0 else (total - exempt) / total
    return percent_of(Decimal(taxable_amount) * share, tax_rate_for(country))
