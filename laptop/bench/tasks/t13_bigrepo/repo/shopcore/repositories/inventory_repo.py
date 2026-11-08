from ..models.inventory_item import InventoryItem  # noqa: F401
from .base import BaseRepository


class InventoryItemRepository(BaseRepository):
    pass


inventory_repo = InventoryItemRepository()
