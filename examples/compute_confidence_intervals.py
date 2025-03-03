# taken from https://github.com/robotic-esp/pdt/blob/main/scripts/python/computeConfidenceInterval.py (but also written by V. H.)
# python translation of the matlab file with the same name

from scipy.stats import binom
import math

import functools


# adding memoization to the binomial cdf
@functools.lru_cache(maxsize=None)
def memoized_binom_cdf(k, n, p):
    return binom.cdf(k, n, p)


# computes the probability, that the median of the true population lies in the
# open interval (i, j)
def computeProbabilityForInterval(n, i, j):
    prob = 1 - memoized_binom_cdf(i - 1, n, 0.5) - memoized_binom_cdf(n - j, n, 0.5)
    return prob


# finds the interval for the median of n samples given a confidence
# probability p
def computeConfidenceInterval(n, p):
    startIndex = 0
    endIndex = n

    if n < 100:  # just brute force it.
        startIndex = 0
        endIndex = n
    elif n < 1000:  # be a little less conservative.
        startIndex = math.ceil(n / 3)
        endIndex = math.ceil(2 * n / 3)
    elif n < 5000:  # be a little less conservative still.
        startIndex = math.ceil(6 * n / 15)
        endIndex = math.ceil(9 * n / 15)
    elif n < 20000:  # be a little less conservative still.
        startIndex = math.ceil(14 * n / 31)
        endIndex = math.ceil(17 * n / 31)
    elif n < 100000:  # be a little less conservative still.
        startIndex = math.ceil(35 * n / 71)
        endIndex = math.ceil(36 * n / 71)
    elif n < 1000000:  # be a little less conservative still.
        startIndex = math.ceil(60 * n / 121)
        endIndex = math.ceil(61 * n / 121)
    elif n == 1000000:  # be a little less conservative still.
        startIndex = math.ceil(170 * n / 341)
        endIndex = math.ceil(171 * n / 341)
    else:
        raise ValueError("Provide bounds, this will take forever.")

    # rint(f'lowest considered idx: {startIndex}, highest considered idx: {endIndex}')

    bestInterval = [startIndex, endIndex]
    bestPercentage = computeProbabilityForInterval(n, startIndex, endIndex)

    if bestPercentage < p:
        return 0, n, bestPercentage
        # raise ValueError("Search is not conservative enough. Lower the startSearchIdx and increase the endSearchIdx.")

    for i in range(
        startIndex, math.floor(n / 2.0) + 1
    ):  # +1, since range() is not inclusive
        for j in range(
            math.ceil(n / 2.0), endIndex + 1
        ):  # +1, since range() is not inclusive
            intervalPercentage = computeProbabilityForInterval(n, i, j)
            if (
                intervalPercentage - p >= 0
                and j - i < bestInterval[1] - bestInterval[0]
            ):
                bestInterval = [i, j]
                bestPercentage = intervalPercentage
                break  # We can safely brake here, any larger j will give a larger interval and higher percentage.

    return bestInterval[0], bestInterval[1], bestPercentage
