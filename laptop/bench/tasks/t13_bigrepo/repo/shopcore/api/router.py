from . import cart_api, customers_api, health, orders_api, payments_api, products_api, reviews_api

ROUTES = {
    ("POST", "/orders"): lambda body, **kw: orders_api.post_order(body),
    ("POST", "/orders/checkout"): lambda body, **kw: orders_api.post_checkout(kw["order_id"], body),
    ("GET", "/products"): lambda body, **kw: products_api.get_products(kw.get("query", {})),
    ("POST", "/cart/items"): lambda body, **kw: cart_api.post_cart_item(kw["customer_id"], body),
    ("POST", "/customers"): lambda body, **kw: customers_api.post_customer(body),
    ("GET", "/payments"): lambda body, **kw: payments_api.get_payments(kw["order_id"]),
    ("POST", "/reviews"): lambda body, **kw: reviews_api.post_review(kw["product_id"], body),
    ("GET", "/health"): lambda body, **kw: health.get_health(),
}


def dispatch(method, path, body=None, **kwargs):
    return ROUTES[(method, path)](body or {}, **kwargs)
