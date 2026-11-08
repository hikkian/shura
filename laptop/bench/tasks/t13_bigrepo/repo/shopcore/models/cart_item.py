from dataclasses import dataclass, field
from decimal import Decimal


@dataclass
class CartItem:
    product_id: int
    quantity: int = 1
