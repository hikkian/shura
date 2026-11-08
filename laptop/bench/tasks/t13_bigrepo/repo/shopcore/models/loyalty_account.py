from dataclasses import dataclass, field
from decimal import Decimal


@dataclass
class LoyaltyAccount:
    customer_id: int
    points: int = 0
