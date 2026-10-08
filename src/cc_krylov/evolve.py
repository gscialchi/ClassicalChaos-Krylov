import numpy as np


def evolve_operator_vN(u, rho, n_final):
    """
    von Neumann evolution of an operator with the unitary
    up to timestep t=n_final.
    """
    if len(rho.shape) > 2: # if True, then it is an evo to continue
        n = rho.shape[0] - 1
    else:
        n = 0
        rho = np.asarray([rho.copy()])

    out = np.zeros((1+n_final, *rho.shape[-2:]), dtype=complex)
    out[:n+1, :, :] = rho[:, :, :]

    rho_t = rho[n, :, :]
    while n < n_final:
        n += 1
        rho_t = u @ rho_t @ u.conj().T
        out[n, :, :] = rho_t
    return out
