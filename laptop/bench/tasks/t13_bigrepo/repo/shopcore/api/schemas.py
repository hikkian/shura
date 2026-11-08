def order_to_dict(order):
    return {"id": order.id, "status": order.status, "price": {k: str(v) for k, v in order.price.items()}}


def product_to_dict(p):
    return {"id": p.id, "name": p.name, "price": str(p.price), "category": p.category}
