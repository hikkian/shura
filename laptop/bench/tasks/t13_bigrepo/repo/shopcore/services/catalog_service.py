from ..repositories.product_repo import product_repo


def list_products(category=None):
    if category:
        return product_repo.find_by_category(category)
    return product_repo.all()
