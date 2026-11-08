"""Правила скидок. Меняйте здесь проценты и пороги — код сервисов трогать не нужно."""

DISCOUNT_RULES = [
    {"name": "volume_10", "type": "volume", "min_items": 10, "percent": 5},
    {"name": "volume_50", "type": "volume", "min_items": 50, "percent": 12},
    {"name": "loyalty_silver", "type": "loyalty", "tier": "silver", "percent": 5},
    {"name": "loyalty_gold", "type": "loyalty", "tier": "gold", "percent": 10},
    {"name": "first_order", "type": "first_order", "percent": 7},
]

# суммарная скидка (правила + купон) не может превышать этот процент
MAX_TOTAL_DISCOUNT_PERCENT = 30
