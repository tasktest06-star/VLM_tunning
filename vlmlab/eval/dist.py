"""Distribution functions, so the statistics layer needs no third-party library.

The only thing in the whole evaluation design that genuinely wants numpy or
scipy is the inverse Student-t, used for the minimum detectable difference.
Approximating it would be the wrong trade when the resulting number decides
whether a result is reportable, so it is implemented properly here from the
regularised incomplete beta function.

The same incomplete beta gives exact binomial intervals, which the research
commits to for per-class recall. One module, two consumers.
"""
import math
from typing import Optional, Tuple

__all__ = [
    "log_beta", "betacf", "betainc", "t_cdf", "t_ppf", "normal_ppf",
    "clopper_pearson", "wilson",
]

_MAXIT = 300
_EPS = 3.0e-16
_FPMIN = 1.0e-300


def log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def betacf(a, b, x):
    """Continued fraction for the incomplete beta, by the modified Lentz method."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b)."""
    if x < 0.0 or x > 1.0:
        raise ValueError("x must be in [0, 1]")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    front = math.exp(math.log(x) * a + math.log1p(-x) * b - log_beta(a, b))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * betacf(a, b, x) / a
    return 1.0 - math.exp(math.log1p(-x) * b + math.log(x) * a
                          - log_beta(b, a)) * betacf(b, a, 1.0 - x) / b


def t_cdf(t, df):
    """Student-t cumulative distribution."""
    if df <= 0:
        raise ValueError("df must be positive")
    x = df / (df + t * t)
    p = 0.5 * betainc(df / 2.0, 0.5, x)
    return 1.0 - p if t > 0 else p


def normal_ppf(p):
    """Inverse standard normal, via the stdlib."""
    from statistics import NormalDist
    return NormalDist().inv_cdf(p)


def t_ppf(p, df, tol=1e-10, max_iter=200):
    """Inverse Student-t by bisection on the cumulative distribution.

    Falls through to the normal for large degrees of freedom, where the two
    agree to well beyond the precision that matters here.
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1)")
    if df <= 0:
        raise ValueError("df must be positive")
    if df > 2000:
        return normal_ppf(p)
    lo, hi = -1e4, 1e4
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial interval. The research commits to this for per-class recall.

    Returns ``(lo, hi)``. Degenerate counts give a one-sided interval rather
    than a nonsensical point.
    """
    if n < 0 or k < 0 or k > n:
        raise ValueError("require 0 <= k <= n")
    if n == 0:
        return (0.0, 1.0)
    a = alpha / 2.0
    lo = 0.0 if k == 0 else _beta_ppf(a, k, n - k + 1)
    hi = 1.0 if k == n else _beta_ppf(1.0 - a, k + 1, n - k)
    return (lo, hi)


def _beta_ppf(p, a, b, tol=1e-12, max_iter=200):
    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if betainc(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def wilson(k, n, alpha=0.05):
    """Wilson score interval. Cheaper than exact, and adequate for monitoring."""
    if n == 0:
        return (0.0, 1.0)
    z = normal_ppf(1.0 - alpha / 2.0)
    phat = k / float(n)
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
