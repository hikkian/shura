from ..services import review_service


def post_review(product_id, body):
    r = review_service.add_review(product_id, body["customer_id"], body["rating"], body.get("text", ""))
    return 201, {"id": r.id}
