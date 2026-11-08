from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")


def quantize(value):
    return Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


def percent_of(amount, percent):
    return quantize(Decimal(amount) * Decimal(percent) / Decimal(100))
