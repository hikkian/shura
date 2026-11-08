from ..models.order import Order
from ..models.order_item import OrderItem
from ..repositories.cart_repo import cart_repo
from ..repositories.coupon_repo import coupon_repo
from ..repositories.customer_repo import customer_repo
from ..repositories.order_repo import order_repo
from ..repositories.product_repo import product_repo
from ..utils.ids import next_id
from . import inventory_service, payment_service, pricing_service


class OrderError(Exception):
    pass


def create_order(customer_id, cart_id, coupon_code=""):
    """Превращает корзину в заказ и фиксирует цену через pricing_service."""
    customer = customer_repo.get(customer_id)
    cart = cart_repo.get(cart_id)
    if customer is None or cart is None or not cart.items:
        raise OrderError("customer or cart not found / cart empty")
    items = []
    for ci in cart.items:
        p = product_repo.get(ci.product_id)
        items.append(OrderItem(p.id, p.name, p.price, ci.quantity, p.category))
    coupon = coupon_repo.get(coupon_code) if coupon_code else None
    first = not order_repo.find_by_customer_id(customer_id)
    order = Order(next_id("order"), customer_id, items, country=customer.country, coupon_code=coupon_code)
    order.price = pricing_service.calculate_final_price(
        items, customer, coupon, customer.country, is_first_order=first
    ).as_dict()
    return order_repo.add(order)


def checkout(order_id, payment_token):
    order = order_repo.get(order_id)
    if order is None:
        raise OrderError("order not found")
    inventory_service.reserve(order.items)
    payment = payment_service.process_payment(order, payment_token)
    if payment.status == "succeeded":
        inventory_service.commit(order.items)
    return order
