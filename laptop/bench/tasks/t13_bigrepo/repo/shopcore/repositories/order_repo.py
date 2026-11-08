from ..models.order import Order  # noqa: F401
from .base import BaseRepository


class OrderRepository(BaseRepository):
    def find_by_customer_id(self, value):
        return [o for o in self._data.values() if o.customer_id == value]

    def find_by_status(self, value):
        return [o for o in self._data.values() if o.status == value]

    pass


order_repo = OrderRepository()
