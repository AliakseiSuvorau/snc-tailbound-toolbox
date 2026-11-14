import inspect
import math
from abc import ABC, abstractmethod
from typing import Callable

from model import Arrival, TokenBucketArrival, DiscreteMMOOArrival
from utils.errors import InvalidParameterError


class Service(ABC):
    @property
    @abstractmethod
    def curve_func(self) -> Callable[..., float]:
        """curve_func must be implemented in subclass"""
        pass

    @property
    @abstractmethod
    def err_func(self) -> Callable[..., float]:
        """err_func must be implemented in subclass"""
        pass


class ConstantRateService(Service):
    """
    Service with stochastic curve \beta = rate * t and envelope P(D(t) < A * [\beta - \\sigma]^+(t)) < err(\\sigma).
    """

    rate: float
    burst: float

    def __init__(self, rate: float, burst: float=0.):
        if rate < 0 or burst < 0:
            raise InvalidParameterError("rate and burst values must be non-negative")

        self.rate = rate
        self.burst = burst

    @property
    def curve_func(self) -> Callable[[int], float]:
        def _curve(t: int) -> float:
            if t <= 0:
                return 0.

            return self.rate * t + self.burst

        return _curve

    @property
    def err_func(self) -> Callable[[float, float], float]:
        return lambda sigma, theta=None: 0.


def _adapt_callable(fn: Callable[..., float]) -> Callable[..., float]:
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


class LeftoverService(Service):
    """
    Stochastic leftover service curve for a flow of interest sharing a
    work-conserving server `service` with cross-traffic `cross_arrival`.

    beta_lo(t) = beta(t) - alpha_{2, slack_rate}(t)
    err_lo(sigma) = sum_{k=0}^{inf} err_{a, 2}(sigma + k * slack_rate)

    Infinite sum for error function is calculated the following way:

    - If cross traffic is deterministic/EBB, the sum will be calculated precisely
    - Otherwise, the sum will be cropped according to either:

        - Precision: If next term of the sum is less than '`atol` + `rtol` * |sum_prefix|'
        - Number of terms: `max_iter`
    """

    service: Service
    cross_arrival: Arrival
    delta: float

    rtol: float
    atol: float
    max_iter: int

    _base_curve: Callable[..., float]
    _cross_curve: Callable[..., float]
    _cross_err: Callable[..., float]

    def __init__(
        self,
        service: Service,
        cross_arrival: Arrival,
        slack_rate: float,
        rtol: float = 1e-12,
        atol: float = 1e-15,
        max_iter: int = 100_000,
    ):
        if slack_rate <= 0:
            raise InvalidParameterError("`slack_rate` must be strictly positive")

        # TODO: Check stability condition (4.22)
        # TODO: Check integrability condition (4.4)

        self.service = service
        self.cross_arrival = cross_arrival
        self.slack_rate = slack_rate
        self.rtol = rtol
        self.atol = atol
        self.max_iter = max_iter

        self._base_curve = _adapt_callable(service.curve_func)
        self._cross_curve = _adapt_callable(cross_arrival.curve_func)
        self._cross_err = _adapt_callable(cross_arrival.err_func)

    @property
    def curve_func(self) -> Callable[..., float]:
        def _curve(t: int, sigma: float, *args: float) -> float:
            if t <= 0:
                return 0.

            beta_t = self._base_curve(t, *args)
            alpha_2_delta_t = self._cross_curve(t, self.slack_rate * t, *args)

            return beta_t - alpha_2_delta_t

        return _curve

    @property
    def err_func(self) -> Callable[..., float]:
        # Deterministic cross-traffic
        if isinstance(self.cross_arrival, TokenBucketArrival):
            return lambda sigma, *args: 0. if sigma >= 0 else 1.

        # EBB cross-traffic
        if isinstance(self.cross_arrival, DiscreteMMOOArrival):
            def _ebb_err(sigma: float, theta: float) -> float:
                if sigma < 0:
                    return 1.
                
                denom = 1.0 - math.exp(-theta * self.slack_rate)
                if denom <= 0.:
                    return 1.

                return min(1., math.exp(-theta * sigma) / denom)

            return _ebb_err

        # General fallback
        def _numeric_err(sigma: float, *args: float) -> float:
            if sigma < 0:
                return 1.

            prefix_sum = 0.
            for k in range(self.max_iter):
                new_term = self._cross_err(sigma + k * self.slack_rate, *args)
                if new_term <= 0.:
                    break

                prefix_sum += new_term

                if prefix_sum >= 1.:
                    return 1.

                if new_term <= self.atol + self.rtol * prefix_sum:
                    break
            else:
                raise InvalidParameterError(
                    f"err_func sum did not converge within {self.max_iter} iterations; "
                    "check integrability condition or increase `delta`."
                )

            return prefix_sum

        return _numeric_err
