from ..models.product import Product  # noqa: F401
from .base import BaseRepository


class ProductRepository(BaseRepository):
    def find_by_category(self, value):
        return [o for o in self._data.values() if o.category == value]

    pass


product_repo = ProductRepository()
