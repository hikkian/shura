from ..services import cart_service


def post_cart_item(customer_id, body):
    cart = cart_service.get_or_create_cart(customer_id)
    cart.add(body["product_id"], body.get("quantity", 1))
    return 200, {"cart_id": cart.id, "preview_total": str(cart_service.cart_preview_total(cart))}
