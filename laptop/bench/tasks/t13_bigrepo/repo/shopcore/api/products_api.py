from ..services import catalog_service
from .schemas import product_to_dict


def get_products(query):
    return 200, [product_to_dict(p) for p in catalog_service.list_products(query.get("category"))]
