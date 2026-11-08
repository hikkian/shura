from .errors import GatewayTimeout, PaymentDeclined


class FakeGateway:
    """Тестовый шлюз: токены tok_declined / tok_timeout имитируют сбои."""

    def charge(self, token, amount):
        if token.startswith("tok_declined"):
            raise PaymentDeclined("card declined")
        if token.startswith("tok_timeout"):
            raise GatewayTimeout("gateway timeout")
        return {"ok": True, "amount": str(amount)}


default_gateway = FakeGateway()
