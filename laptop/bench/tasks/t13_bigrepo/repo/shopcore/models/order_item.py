from dataclasses import dataclass, field
from decimal import Decimal


@dataclass
class OrderItem:
    product_id: int
    name: str
    unit_price: Decimal
    quantity: int = 1
    category: str = 'general'
