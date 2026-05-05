import time
from contextlib import ContextDecorator


def _emit(scope: str, message: str) -> None:
    print(f"[TIMING][{scope}] {message}", flush=True)


def timing_point(scope: str, message: str) -> None:
    _emit(scope, message)


class _TimedStage(ContextDecorator):
    def __init__(self, scope: str, label: str):
        self.scope = scope
        self.label = label
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        _emit(self.scope, f"start {self.label}")
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self.start_time
        if exc_type is None:
            _emit(self.scope, f"{self.label} done in {elapsed:.3f}s")
        else:
            _emit(self.scope, f"{self.label} failed after {elapsed:.3f}s: {exc_type.__name__}")
        return False


def timed_stage(scope: str, label: str) -> _TimedStage:
    return _TimedStage(scope, label)
