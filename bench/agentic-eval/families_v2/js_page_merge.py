"""Catalog entry for an independent v2 task family."""
from .catalog import FAMILIES
FAMILY = next(item for item in FAMILIES if item["name"] == 'js_page_merge')
