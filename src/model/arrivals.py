import math
from abc import ABC, abstractmethod
from typing import Callable
from warnings import warn

from utils.errors import InvalidParameterError
from utils.utils import adapt_callable


class Arrival(ABC):
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


class TokenBucketArrival(Arrival):
    """
    Token Bucket arrival process.
    """

    rate: float
    burst: float

    def __init__(self, rate: float, burst: float):
        if rate < 0 or burst < 0:
            raise InvalidParameterError("rate and burst values must be non-negative")

        self.rate = rate
        self.burst = burst

    @property
    def curve_func(self) -> Callable[[int, float, float], float]:
        def _curve(t, sigma, *args):
            if t <= 0:
                return 0.

            return self.rate * t + self.burst + sigma

        return _curve

    @property
    def err_func(self) -> Callable[[float], float]:
        return lambda sigma: 0. if sigma >= 0 else 1.


class DiscreteMMOOArrival(Arrival):
    """
    Discrete Markov-Modulated On-Off arrival process with exponentially-bounded burstiness.
    Starts at the steady state of the distribution.
    """

    stay_on: float
    stay_off: float
    peak_rate: float

    def __init__(self, stay_on: float, stay_off: float, peak_rate: float):
        if stay_on < 0 or stay_off < 0 or stay_on > 1 or stay_off > 1:
            raise InvalidParameterError("probabilities `stay_on` and `stay_off` must lie in [0,1]")
        if stay_on == 0 or stay_off == 0 or stay_on == 1 or stay_off == 1:
            warn("for absorbing-OFF and -ON MMOO process better use `TokenBucketArrival")
        self.stay_on = stay_on
        self.stay_off = stay_off

        if peak_rate < 0:
            raise InvalidParameterError("arrival rate must be positive")
        if peak_rate == 0:
            warn("for zero-rate process better use `TokenBucketArrival`")
        self.peak_rate = peak_rate

    @property
    def curve_func(self) -> Callable[[int, float, float], float]:
        return lambda t, sigma, theta: self._rate(theta) * t + self._burst(theta) + sigma

    @property
    def err_func(self) -> Callable[[float], float]:
        return lambda sigma, theta: math.exp(-theta * sigma)

    def _rate(self, theta: float) -> float:
        off_on = self.stay_off + self.stay_on * math.exp(theta * self.peak_rate)

        inner_sqrt = 1 - 4 * (self.stay_off + self.stay_on - 1) * math.exp(theta * self.peak_rate) / (off_on * off_on)
        sqrt_part = math.sqrt(max(0., inner_sqrt))

        spectral_rad = 0.5 * off_on * (1 + sqrt_part)

        rho = math.log(spectral_rad) / theta

        if rho < 0:
            raise InvalidParameterError("arrival rate must be non-negative")

        return rho

    def _burst(self, theta: float) -> float:
        off_on = self.stay_off + self.stay_on * math.exp(theta * self.peak_rate)
        sqrt_part = math.sqrt(off_on ** 2 - 4 * (self.stay_off + self.stay_on - 1) * math.exp(theta * self.peak_rate))

        spectral_rad = 0.5 * (off_on + sqrt_part)

        eigen_vec = [1 - self.stay_off, spectral_rad - self.stay_off]

        if eigen_vec[0] * eigen_vec[1] > 0:
            if eigen_vec[0] > 0:
                factor = max(eigen_vec) / min(eigen_vec)
            else:
                factor = min(eigen_vec) / max(eigen_vec)
        else:
            raise InvalidParameterError(f"Markov chain is not ergodic: check transition probabilities.")

        return math.log(math.exp(theta * self.peak_rate) * factor / spectral_rad) / theta


class AggregatedArrival(Arrival):
    """
    Aggregated arrival curve for a sum of `arrivals`.
    """

    arrivals: list[Arrival]

    _adopted_curves: list[Callable[..., float]]
    _adopted_errs: list[Callable[..., float]]

    def __init__(self, arrivals: list[Arrival]):
        if not arrivals:
            raise InvalidParameterError("No arrivals provided")
        if len(arrivals) == 1:
            warn("Better use a single arrival class instead")
        if len(arrivals) != len(set(arrivals)):
            raise InvalidParameterError("Arrivals cannot repeat")
        self.arrivals = arrivals

        self._adopted_curves = [adapt_callable(arrival.curve_func) for arrival in arrivals]
        self._adopted_errs = [adapt_callable(arrival.err_func) for arrival in arrivals]

    @property
    def curve_func(self) -> Callable[..., float]:
        return lambda t, sigma, *args: sum(fn(t, sigma, *args) for fn in self._adopted_curves)

    @property
    def err_func(self) -> Callable[..., float]:
        return lambda sigma, *args: min(1., sum(fn(sigma, *args) for fn in self._adopted_errs))
