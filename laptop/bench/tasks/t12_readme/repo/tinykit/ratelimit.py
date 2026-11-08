import time


class RateLimitExceeded(Exception):
    def __init__(self, retry_after):
        super().__init__(f"rate limit exceeded, retry after {retry_after:.2f}s")
        self.retry_after = retry_after


class TokenBucket:
    def __init__(self, rate, capacity, clock=time.monotonic):
        if rate <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be positive")
        self.rate = float(rate)
        self.capacity = float(capacity)
        self._clock = clock
        self._tokens = float(capacity)
        self._last = clock()

    def _refill(self):
        now = self._clock()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
        self._last = now

    def try_acquire(self, n=1):
        self._refill()
        if n <= self._tokens:
            self._tokens -= n
            return True
        return False

    def acquire(self, n=1):
        if not self.try_acquire(n):
            raise RateLimitExceeded(self.wait_time(n))

    def wait_time(self, n=1):
        self._refill()
        missing = n - self._tokens
        return 0.0 if missing <= 0 else missing / self.rate
