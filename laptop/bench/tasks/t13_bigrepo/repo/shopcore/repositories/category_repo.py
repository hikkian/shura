from ..models.category import Category  # noqa: F401
from .base import BaseRepository


class CategoryRepository(BaseRepository):
    def find_by_slug(self, value):
        return [o for o in self._data.values() if o.slug == value]

    pass


category_repo = CategoryRepository()
