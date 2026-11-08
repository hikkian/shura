from dataclasses import dataclass, field
from decimal import Decimal


@dataclass
class InventoryItem:
    product_id: int
    on_hand: int = 0
    reserved: int = 0
