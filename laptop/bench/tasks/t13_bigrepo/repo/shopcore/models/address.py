from dataclasses import dataclass, field
from decimal import Decimal


@dataclass
class Address:
    country: str = 'KZ'
    city: str = ''
    street: str = ''
    zip_code: str = ''
