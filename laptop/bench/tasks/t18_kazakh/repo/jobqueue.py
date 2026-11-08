"""In-memory job queue with retries and exponential backoff."""
import heapq
import itertools


def backoff_delay(attempt, base=2, cap=60):
    """Delay (seconds) before the next retry after `attempt` failed attempts."""
    return min(cap, base ** attempt)


class Job:
    def __init__(self, job_id, func, max_attempts=3):
        self.id = job_id
        self.func = func
        self.max_attempts = max_attempts
        self.attempts = 0
        self.status = "pending"  # pending | done | dead
        self.last_error = None


class JobQueue:
    def __init__(self):
        self._heap = []              # (run_at, seq, job)
        self._seq = itertools.count()
        self._ids = itertools.count(1)
        self.dead_letter = []        # jobs that exhausted all attempts
        self.results = {}

    def enqueue(self, func, max_attempts=3, run_at=0.0):
        job = Job(next(self._ids), func, max_attempts)
        heapq.heappush(self._heap, (run_at, next(self._seq), job))
        return job

    def run_next(self, now):
        """Run the earliest job that is due at time `now`. Returns the job or None."""
        if not self._heap or self._heap[0][0] > now:
            return None
        _, _, job = heapq.heappop(self._heap)
        job.attempts += 1
        try:
            self.results[job.id] = job.func()
            job.status = "done"
        except Exception as exc:  # noqa: BLE001
            job.last_error = repr(exc)
            if job.attempts >= job.max_attempts:
                job.status = "dead"
                self.dead_letter.append(job)
            else:
                run_at = now + backoff_delay(job.attempts)
                heapq.heappush(self._heap, (run_at, next(self._seq), job))
        return job

    def stats(self):
        return {"pending": len(self._heap), "dead": len(self.dead_letter), "done": len(self.results)}
