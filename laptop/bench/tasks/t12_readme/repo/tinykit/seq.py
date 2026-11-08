def chunked(iterable, size):
    if size < 1:
        raise ValueError("size must be >= 1")
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf
