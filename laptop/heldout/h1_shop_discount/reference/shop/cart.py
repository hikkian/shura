from decimal import ROUND_HALF_UP, Decimal

from .money import to_cents


class Cart:
    def __init__(self):
        self.items = []  # (name, unit_cents, qty)

    def add(self, name, price, qty=1):
        """price: строка ("19,99") или int копеек."""
        cents = price if isinstance(price, int) else to_cents(price)
        if qty < 1:
            raise ValueError("qty must be >= 1")
        self.items.append((name, cents, qty))

    def subtotal(self):
        return sum(c * q for _, c, q in self.items)

    def discount(self, pct):
        """Сумма скидки в копейках для pct процентов (0..100)."""
        p = Decimal(str(pct))
        if not (0 <= p <= 100):
            raise ValueError("pct must be in 0..100")
        return int((Decimal(self.subtotal()) * p / 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))

    def total(self, pct=0):
        return self.subtotal() - self.discount(pct)
