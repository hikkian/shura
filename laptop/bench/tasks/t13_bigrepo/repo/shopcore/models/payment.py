from dataclasses import dataclass, field
from decimal import Decimal


@dataclass
class Payment:
    id: int
    order_id: int
    amount: Decimal
    status: str = 'pending'
    attempts: int = 0
    error: str = ''
