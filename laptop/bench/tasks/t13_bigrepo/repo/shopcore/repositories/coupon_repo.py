from ..models.coupon import Coupon  # noqa: F401
from .base import BaseRepository


class CouponRepository(BaseRepository):
    pass


coupon_repo = CouponRepository()
