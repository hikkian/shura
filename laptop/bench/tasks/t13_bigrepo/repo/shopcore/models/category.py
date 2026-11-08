from dataclasses import dataclass, field
from decimal import Decimal


@dataclass
class Category:
    id: int
    name: str
    slug: str
