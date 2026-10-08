import numpy as np

from cc_krylov.products import operator_norm


def autocorrelation(evo, hbar):
    """
    Autocorrelation for the evolution of an operator, with the
    `operator_prod` product defined in products.py
    """
    evo_normed = evo/operator_norm(evo[0], hbar=hbar)
    out = np.einsum('ki,tik->t', evo_normed[0], evo_normed)
    out /= (2*np.pi*hbar)
    return out
