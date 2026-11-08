from dataclasses import dataclass, field
from decimal import Decimal


@dataclass
class Invoice:
    id: int
    order_id: int
    total: Decimal
    lines: list
