from ..models.loyalty_account import LoyaltyAccount
from ..config.feature_flags import is_enabled

_accounts = {}


def add_points(customer_id, order_total):
    if not is_enabled("loyalty_points"):
        return 0
    acc = _accounts.setdefault(customer_id, LoyaltyAccount(customer_id))
    acc.points += int(order_total // 100)
    return acc.points
