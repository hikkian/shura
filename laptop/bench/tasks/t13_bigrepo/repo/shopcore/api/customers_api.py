from ..models.customer import Customer
from ..repositories.customer_repo import customer_repo
from ..utils.ids import next_id
from ..utils.validators import is_valid_email
from .errors import ApiError


def post_customer(body):
    if not is_valid_email(body.get("email")):
        raise ApiError(422, "invalid email")
    c = customer_repo.add(Customer(next_id("customer"), body["name"], body["email"]))
    return 201, {"id": c.id}
