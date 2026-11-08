from dataclasses import dataclass, field
from decimal import Decimal


@dataclass
class Customer:
    id: int
    name: str
    email: str
    tier: str = 'regular'
    country: str = 'KZ'
