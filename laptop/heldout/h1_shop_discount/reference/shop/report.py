from .money import fmt


def render(cart, pct=0):
    """Текстовый отчёт по корзине."""
    lines = []
    for name, cents, qty in cart.items:
        lines.append(f"{name} x{qty}: {fmt(cents * qty)}")
    if pct:
        lines.append(f"Скидка {pct}%: {fmt(-cart.discount(pct))}")
    lines.append(f"Итого: {fmt(cart.total(pct))}")
    return "\n".join(lines)
