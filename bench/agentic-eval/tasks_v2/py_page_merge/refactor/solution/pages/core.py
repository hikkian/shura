def merge_pages(pages, key):
    seen, result = set(), []
    for page in pages:
        for item in page:
            value = item[key]
            if value not in seen:
                seen.add(value); result.append(item)
    return sorted(result, key=lambda item: item[key])
