def paginate(items, page, per_page):
    """Return items for 1-based page number. Empty list if page is out of range."""
    if page < 1 or per_page < 1:
        raise ValueError("page and per_page must be >= 1")
    start = page * per_page
    return items[start:start + per_page]


def page_count(total, per_page):
    return total // per_page
