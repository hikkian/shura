from ..models.payment import Payment  # noqa: F401
from .base import BaseRepository


class PaymentRepository(BaseRepository):
    def find_by_order_id(self, value):
        return [o for o in self._data.values() if o.order_id == value]

    pass


payment_repo = PaymentRepository()
