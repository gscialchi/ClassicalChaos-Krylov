import numpy as np


def periodic_gauss_1D(x, m, s):
    """
    Periodic Gaussian distribution with mean position `m` and
    standard deviation `s`, defined on the unit interval.
    """
    out = np.zeros_like(x)
    for n in range(-10, 10): # sum for periodization. Larger range for larger `s`
        out += np.exp(-(x-m+n)**2/(2*s**2))
    out /= (2*np.pi*s**2)**0.5 # same normalization as usual Gaussian holds
    return out


def periodic_gauss_2D(x, y, x0, y0, s):
    """
    Periodic 2D Gaussian distribution with mean position `m` and
    standard deviation `s`, defined on the unit square.
    """
    return periodic_gauss_1D(x, x0, s)*periodic_gauss_1D(y, y0, s)
