from ..models.cart import Cart  # noqa: F401
from .base import BaseRepository


class CartRepository(BaseRepository):
    def find_by_customer_id(self, value):
        return [o for o in self._data.values() if o.customer_id == value]

    pass


cart_repo = CartRepository()
