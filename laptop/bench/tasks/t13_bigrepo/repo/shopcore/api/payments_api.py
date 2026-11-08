from ..repositories.payment_repo import payment_repo


def get_payments(order_id):
    return 200, [{"id": p.id, "status": p.status, "attempts": p.attempts} for p in payment_repo.find_by_order_id(order_id)]
