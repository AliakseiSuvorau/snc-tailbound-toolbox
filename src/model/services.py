from abc import ABC, abstractmethod
from typing import Callable

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
    def curve_func(self) -> Callable[[int, float], float]:
        return lambda t, sigma, theta=None: max(0., self.rate * t + self.burst - sigma)

    @property
    def err_func(self) -> Callable[[float, float], float]:
        return lambda sigma, theta=None: 0.
