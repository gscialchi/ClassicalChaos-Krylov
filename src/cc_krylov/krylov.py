import numpy as np

from cc_krylov.products import *


def arnoldi_FO_operator(U: np.ndarray,
                        e0: np.ndarray,
                        n_final: int,
                        prod = frobenius_prod,
                        ):
    """
    Full re-orthonormalization Arnoldi algorithm for operators.
    """
    def L(op): return U @ op @ U.conj().T
    max_stop = U.shape[0]**2 - U.shape[0] + 1

    def norm(e): return np.sqrt(np.real(prod(e, e)))

    if len(e0.shape) > 2: # if True, then it is an iteration to continue
        n = e0.shape[0] - 1
    else:
        n = 0
        e0 = np.asarray([e0.copy()/norm(e0)])

    if (U.dtype == complex) or (e0.dtype == complex):
        e_arr = np.zeros((1+n_final, *e0.shape[-2:]), dtype=complex)
    else:
        e_arr = np.zeros((1+n_final, *e0.shape[-2:]))
    e_arr[:n+1, :, :] = e0[:, :, :]

    bn = np.inf
    kn = e_arr[n, :, :]
    while (bn > 1e-7) and (n < n_final):
        n += 1
        An = L(kn)
        for _ in range(2):
            for km in e_arr[:n+1]:
                An -= km * prod(km, An)
        bn = norm(An)
        kn = An/bn
        e_arr[n, :, :] = kn
    print(f'Stopped at n={n} with bn={bn}. Max. possible stop={max_stop}.')
    return e_arr[:n+1] # remove trailing zeros if stops early


def krylov_propagator_kry(u, kry, hbar):
    """
    Compute the matrix representation of the propagator in the Krylov basis
    from the Krylov states.
    """
    #  nk = len(kry)
    #  out = np.zeros((nk, nk))

    #  def L(op): return u @ op @ u.conj().T

    #  for m in range(nk):
        #  km = kry[m]
        #  for n in range(m-1, nk):
            #  kn = kry[n]
            #  out[m, n] = prod(km, L(kn))

    # the computation below is equivalent to the above
    kry_dag = np.transpose(kry.conj(), axes=(0, 2, 1)) # dag every k inside
    out = np.einsum('mab,bi,nij,ja->mn', kry_dag, u, kry, u.conj().T,
                    optimize=True)/(2*np.pi*hbar)
    # this assumes the `operator_prod` product defined in products.py
    return out


def arnoldi_sequences_verblunsky(u, krylov, hbar):
    """
    Get Arnoldi sequences & Verblunsky coefficients from the unitary.
    """
    u_krylov = krylov_propagator_kry(u, krylov, hbar)
    an = np.diag(u_krylov, k=0)
    bn = np.diag(u_krylov, k=-1) # n >= 1
    cn = u_krylov[0, :]
    vn = (1 - bn**2)**0.5
    return an, bn, cn, vn
