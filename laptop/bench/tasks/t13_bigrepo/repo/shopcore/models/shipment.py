from dataclasses import dataclass, field
from decimal import Decimal


@dataclass
class Shipment:
    id: int
    order_id: int
    cost: Decimal
    status: str = 'new'
