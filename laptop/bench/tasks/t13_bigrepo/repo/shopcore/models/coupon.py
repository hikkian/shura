from dataclasses import dataclass, field
from decimal import Decimal


@dataclass
class Coupon:
    code: str
    percent: int = 0
    active: bool = True
    max_uses: int = 100
    used: int = 0
