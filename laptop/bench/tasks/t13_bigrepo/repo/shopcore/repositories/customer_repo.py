from ..models.customer import Customer  # noqa: F401
from .base import BaseRepository


class CustomerRepository(BaseRepository):
    def find_by_email(self, value):
        return [o for o in self._data.values() if o.email == value]

    pass


customer_repo = CustomerRepository()
