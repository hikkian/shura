from ..models.cart import Cart
from ..repositories.cart_repo import cart_repo
from ..repositories.product_repo import product_repo
from ..utils.ids import next_id
from . import pricing_service


def get_or_create_cart(customer_id):
    existing = cart_repo.find_by_customer_id(customer_id)
    return existing[0] if existing else cart_repo.add(Cart(next_id("cart"), customer_id))


def cart_preview_total(cart):
    """Ориентировочная сумма корзины без скидок и налога (показывается до оформления)."""
    from ..models.order_item import OrderItem
    items = []
    for ci in cart.items:
        p = product_repo.get(ci.product_id)
        items.append(OrderItem(p.id, p.name, p.price, ci.quantity, p.category))
    return pricing_service.calculate_subtotal(items)
