from ..models.review import Review  # noqa: F401
from .base import BaseRepository


class ReviewRepository(BaseRepository):
    def find_by_product_id(self, value):
        return [o for o in self._data.values() if o.product_id == value]

    pass


review_repo = ReviewRepository()
