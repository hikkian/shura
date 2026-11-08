from dataclasses import dataclass, field
from decimal import Decimal


@dataclass
class Review:
    id: int
    product_id: int
    customer_id: int
    rating: int = 5
    text: str = ''
