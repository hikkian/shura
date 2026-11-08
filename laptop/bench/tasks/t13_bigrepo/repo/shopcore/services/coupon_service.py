from ..repositories.coupon_repo import coupon_repo


def validate_coupon(code):
    c = coupon_repo.get(code)
    return c is not None and c.active and c.used < c.max_uses


def mark_used(code):
    c = coupon_repo.get(code)
    if c:
        c.used += 1
        if c.used >= c.max_uses:
            c.active = False
