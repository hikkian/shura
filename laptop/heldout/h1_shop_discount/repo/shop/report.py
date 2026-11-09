from .money import fmt


def render(cart):
    """Текстовый отчёт по корзине."""
    lines = []
    for name, cents, qty in cart.items:
        lines.append(f"{name} x{qty}: {fmt(cents * qty)}")
    lines.append(f"Итого: {fmt(cart.subtotal())}")
    return "\n".join(lines)
