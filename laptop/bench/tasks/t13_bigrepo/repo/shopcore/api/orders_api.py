from ..services import order_service
from .errors import ApiError
from .schemas import order_to_dict


def post_order(body):
    try:
        order = order_service.create_order(body["customer_id"], body["cart_id"], body.get("coupon", ""))
    except order_service.OrderError as e:
        raise ApiError(400, str(e))
    return 201, order_to_dict(order)


def post_checkout(order_id, body):
    order = order_service.checkout(order_id, body["payment_token"])
    return 200, order_to_dict(order)
