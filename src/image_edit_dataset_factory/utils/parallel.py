from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import TypeVar

from tqdm import tqdm

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    num_workers: int = 4,
    use_processes: bool = False,
    desc: str | None = None,
) -> list[R]:
    values = list(items)
    if num_workers <= 1:
        return [fn(item) for item in tqdm(values, desc=desc)]

    executor_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    results: list[R] = []
    with executor_cls(max_workers=num_workers) as executor:
        futures = [executor.submit(fn, item) for item in values]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            results.append(fut.result())
    return results
