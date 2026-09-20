"""Bounded ordered CPU-only I/O. Never executes models or decision rules."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor


class OrderedPrefetch:
    def __init__(self, function, iterable, workers):
        self.source = iter(iterable)
        self.function = function
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='b5_read')
        self.pending = deque()
        self.closed = False
        try:
            for _ in range(workers):
                if not self._submit():
                    break
        except BaseException:
            self.close()
            raise

    def _submit(self):
        try:
            item = next(self.source)
        except StopIteration:
            return False
        self.pending.append(self.pool.submit(self.function, item))
        return True

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed or not self.pending:
            self.close()
            raise StopIteration
        try:
            result = self.pending.popleft().result()
            self._submit()
            return result
        except BaseException:
            self.close()
            raise

    def close(self):
        if not self.closed:
            self.closed = True
            for future in self.pending:
                future.cancel()
            self.pool.shutdown(wait=True)  # Python 3.8 compatible
            self.pending.clear()
