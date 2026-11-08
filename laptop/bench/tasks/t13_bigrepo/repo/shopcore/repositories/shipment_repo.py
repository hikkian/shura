from ..models.shipment import Shipment  # noqa: F401
from .base import BaseRepository


class ShipmentRepository(BaseRepository):
    def find_by_order_id(self, value):
        return [o for o in self._data.values() if o.order_id == value]

    pass


shipment_repo = ShipmentRepository()
