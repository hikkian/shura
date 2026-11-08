from ..models.review import Review
from ..repositories.review_repo import review_repo
from ..utils.ids import next_id


def add_review(product_id, customer_id, rating, text=""):
    if not 1 <= rating <= 5:
        raise ValueError("rating 1..5")
    return review_repo.add(Review(next_id("review"), product_id, customer_id, rating, text))


def average_rating(product_id):
    rs = review_repo.find_by_product_id(product_id)
    return sum(r.rating for r in rs) / len(rs) if rs else None
