from ..repositories.inventory_repo import inventory_repo
from ..models.inventory_item import InventoryItem


class OutOfStock(Exception):
    pass


def restock(product_id, qty):
    item = inventory_repo.get(product_id) or inventory_repo.add(InventoryItem(product_id), key=product_id)
    item.on_hand += qty


def reserve(items):
    for it in items:
        rec = inventory_repo.get(it.product_id)
        if rec is None or rec.on_hand - rec.reserved < it.quantity:
            raise OutOfStock(it.product_id)
    for it in items:
        inventory_repo.get(it.product_id).reserved += it.quantity


def release(items):
    for it in items:
        rec = inventory_repo.get(it.product_id)
        if rec:
            rec.reserved = max(0, rec.reserved - it.quantity)


def commit(items):
    for it in items:
        rec = inventory_repo.get(it.product_id)
        rec.on_hand -= it.quantity
        rec.reserved -= it.quantity
