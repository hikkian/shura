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
        return round(self.subtotal() * pct / 100)

    def total(self, pct=0):
        return self.subtotal() - self.discount(pct)
