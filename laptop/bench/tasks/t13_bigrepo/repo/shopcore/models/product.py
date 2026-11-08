from dataclasses import dataclass, field
from decimal import Decimal


@dataclass
class Product:
    id: int
    name: str
    price: Decimal
    category: str = 'general'
    weight_kg: float = 0.5
