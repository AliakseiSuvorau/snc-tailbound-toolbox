import inspect
from typing import Callable


def adapt_callable(fn: Callable[..., float]) -> Callable[..., float]:
    """
    Wraps `fn` so extra trailing positional args (like `theta`) are only passed
    if `fn` actually accepts them.
    """
    params = inspect.signature(fn).parameters.values()

    # Function signature includes *args
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        return fn

    # If a function fn accepts fewer parameters than passed, we trim parameters list
    max_positional = sum(
        1 for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    return lambda *args: fn(*args[:max_positional])
