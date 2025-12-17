from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class TimedBlock:
    start_s: float
    end_s: float

    @property
    def duration_ms(self) -> float:
        return (self.end_s - self.start_s) * 1000.0


@contextmanager
def timed_block() -> TimedBlock:
    start = time.perf_counter()
    try:
        yield TimedBlock(start_s=start, end_s=start)
    finally:
        end = time.perf_counter()
        # Consumers read the returned object, so update via local variable is not possible.
        # Provide a new TimedBlock through return value in helpers instead.


def time_call(fn, *args, **kwargs) -> tuple[TimedBlock, object]:
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    end = time.perf_counter()
    return TimedBlock(start_s=start, end_s=end), out

