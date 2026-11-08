import unittest
from decimal import Decimal

from shopcore.models.cart import Cart
from shopcore.models.coupon import Coupon
from shopcore.models.customer import Customer
from shopcore.models.product import Product
from shopcore.payments.failure_handler import handle_payment_failure  # noqa: F401
from shopcore.repositories.cart_repo import cart_repo
from shopcore.repositories.coupon_repo import coupon_repo
from shopcore.repositories.customer_repo import customer_repo
from shopcore.repositories.inventory_repo import inventory_repo
from shopcore.repositories.order_repo import order_repo
from shopcore.repositories.product_repo import product_repo
from shopcore.services import inventory_service, order_service
from shopcore.tasks.retry_payments import SCHEDULED
from shopcore.utils.ids import reset_ids


class FlowTest(unittest.TestCase):
    def setUp(self):
        for r in (cart_repo, coupon_repo, customer_repo, inventory_repo, order_repo, product_repo):
            r.clear()
        SCHEDULED.clear()
        reset_ids()
        product_repo.add(Product(1, "Ноутбук", Decimal("100000")))
        product_repo.add(Product(2, "Книга", Decimal("5000"), category="books"))
        inventory_service.restock(1, 10)
        inventory_service.restock(2, 10)
        customer_repo.add(Customer(1, "Асель", "a@mail.kz", tier="gold"))
        cart = cart_repo.add(Cart(1, 1))
        cart.add(1, 1)
        cart.add(2, 2)

    def test_price_breakdown(self):
        o = order_service.create_order(1, 1)
        # подытог 110000; скидка gold 10% + первый заказ 7% = 17% = 18700; после скидки 91300
        self.assertEqual(o.price["subtotal"], Decimal("110000.00"))
        self.assertEqual(o.price["discount_percent"], 17)
        self.assertEqual(o.price["discount"], Decimal("18700.00"))
        # налог 12% только с неосвобождённой части (100000/110000 от 91300)
        self.assertEqual(o.price["total"], Decimal("91300.00") + o.price["tax"])

    def test_coupon_cap(self):
        coupon_repo.add(Coupon("BIG", percent=40), key="BIG")
        o = order_service.create_order(1, 1, "BIG")
        self.assertEqual(o.price["discount_percent"], 30)

    def test_checkout_ok(self):
        o = order_service.create_order(1, 1)
        order_service.checkout(o.id, "tok_ok")
        self.assertEqual(o.status, "paid")
        self.assertEqual(inventory_repo.get(1).on_hand, 9)

    def test_checkout_declined(self):
        o = order_service.create_order(1, 1)
        order_service.checkout(o.id, "tok_declined")
        self.assertEqual(o.status, "payment_failed")
        self.assertEqual(inventory_repo.get(1).reserved, 0)
        self.assertEqual(SCHEDULED, [])

    def test_checkout_timeout_schedules_retry(self):
        o = order_service.create_order(1, 1)
        order_service.checkout(o.id, "tok_timeout")
        self.assertEqual(o.status, "payment_failed")
        self.assertEqual(len(SCHEDULED), 1)


if __name__ == "__main__":
    unittest.main()
