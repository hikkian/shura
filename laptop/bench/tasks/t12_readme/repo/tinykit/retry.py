import functools
import time


def retry(times=3, delay=0.0, exceptions=(Exception,), sleep=time.sleep):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last = None
            for attempt in range(times):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last = e
                    if attempt < times - 1 and delay:
                        sleep(delay * (2 ** attempt))
            raise last
        return wrapper
    return deco
