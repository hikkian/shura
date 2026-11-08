from dataclasses import dataclass, field

from .cart_item import CartItem


@dataclass
class Cart:
    id: int
    customer_id: int
    items: list = field(default_factory=list)

    def add(self, product_id, quantity=1):
        for it in self.items:
            if it.product_id == product_id:
                it.quantity += quantity
                return
        self.items.append(CartItem(product_id, quantity))
