from itertools import combinations
from multiprocessing import Pool

import numpy as np
from scipy.integrate import simpson


PI = np.pi
DPI = 2*np.pi


### Classical maps
def c_cat_perturbed(k, q, p):
    """
    Classical perturbed Arnold's cat map in the unit torus.
    """
    p_ = p + q - DPI*k*np.sin(DPI*q)
    q_ = q + p_ + DPI*k*np.sin(DPI*p_)
    return np.mod(q_, 1), np.mod(p_, 1)


def c_harper(k, q, p):
    """
    Classical Harper map on the unit torus.
    """
    p_ = p - k*np.sin(DPI*q)
    q_ = q + k*np.sin(DPI*p_)
    return np.mod(q_, 1), np.mod(p_, 1)


def c_standard(k, q, p):
    """
    Classical Standard map on the unit torus.
    """
    p_ = p + k/DPI * np.sin(DPI*q)
    q_ = q + p_
    return np.mod(q_, 1), np.mod(p_, 1)


## Jacobians
def jacobian_cat_perturbed(k, q, p):
    p_ = p + q - DPI*k*np.sin(DPI*q)
    cp_ = (DPI)**2 * k * np.cos(DPI*p_)

    dpp = 1
    dpq = 1 - (DPI)**2 * k * np.cos(DPI*q)
    dqp = 1 + cp_
    dqq = 1 + (1 + cp_)*dpq
    return np.asarray([[dpp, dpq], [dqp, dqq]])


def jacobian_harper(k, q, p):
    p_ = p - k*np.sin(DPI*q)

    dpp = 1
    dpq = -DPI*k*np.cos(DPI*q)
    dqp = DPI*k*np.cos(DPI*p_)
    dqq = 1 + DPI*k*np.cos(DPI*p_)*dpq
    return np.asarray([[dpp, dpq], [dqp, dqq]])


def jacobian_standard(k, q, p):
    dpp = 1
    dpq = k*np.cos(DPI*q)
    dqp = 1
    dqq = 1 + dpq
    return np.asarray([[dpp, dpq], [dqp, dqq]])


### Quantum maps
"""
Globally set the Floquet angles explicitly.
"""
QBAR, PBAR = (0, 0)


def q_harper(k, N, qbar=QBAR, pbar=PBAR):
    """
    Unitary for the quantum Harper map.
    """
    qn = np.arange(0, N) + qbar # n + pbar
    pn = np.arange(0, N) + pbar # m + qbar

    qq, pp = np.meshgrid(qn, pn)
    up = np.exp(1j*N*k*np.cos(DPI/N * pp))
    uq = np.exp(1j*N*k*np.cos(DPI/N * qq))
    ff = np.exp(1j * DPI/N * qq * pp) / N**0.5
    return (ff * up).T @ (ff.conj() * uq)


def q_standard(k, N, qbar=QBAR, pbar=PBAR):
    """
    Unitary for the quantum Standard map.
    """
    qn = np.arange(0, N) + qbar # n + pbar
    pn = np.arange(0, N) + pbar # m + qbar

    qq, pp = np.meshgrid(qn, pn)
    up = np.exp(-1j * PI/N * pp**2)
    uq = np.exp(-1j*N*(k/DPI)*np.cos(DPI/N * qq) )
    ff = np.exp(1j * DPI/N * qq * pp) / N**0.5
    return (ff * up).T @ (ff.conj() * uq)


def q_cat_perturbed(k, N, qbar=QBAR, pbar=PBAR):
    """
    Unitary for the perturbed quantum Arnold's cat map.
    """
    qn = np.arange(0, N) + qbar # n + pbar
    pn = np.arange(0, N) + pbar # m + qbar

    qq, pp = np.meshgrid(qn, pn)
    up = np.exp(-1j*DPI*( pp**2/(2*N) - k*N*np.cos(DPI/N * pp) ))
    uq = np.exp(1j*DPI*( qq**2/(2*N) + k*N*np.cos(DPI/N * qq) ))
    ff = np.exp(1j * DPI/N * qq * pp) / N**0.5
    return (ff * up).T @ (ff.conj() * uq)


##
def jacobi_theta(q, p, N, qn):
    """
    Approximated controlled Jacobi Theta function by selecting a good cutoff
    for the sum. It is accurate to within an atol of at least 1e-9/N
    for all q, p, qn in the range [0, 1].

    See: https://github.com/gscialchi/QtoC-Krylov/tree/main/docs/docs.pdf.
    """
    dq = q - qn
    piN = PI*N
    ord0 = np.exp(-piN*dq**2)
    ord1 = np.exp(-piN*(dq - 1)**2 + 2j*piN*p) + np.exp(-piN*(dq + 1)**2 - 2j*piN*p)
    ord2 = np.exp(-piN*(dq - 2)**2 + 4j*piN*p) + np.exp(-piN*(dq + 2)**2 - 4j*piN*p)
    return ord0 + ord1 + ord2


def coherent_ensemble_torus(f, N, qbar=QBAR, args=()):
    """
    Density matrix for a classical ensemble of coherent states on the unit
    torus written in the position representation.

    See: https://github.com/gscialchi/QtoC-Krylov/tree/main/docs/docs.pdf.
    """
    qlims = [0, 1]
    plims = [0, 1]
    qn = (np.arange(0, N) + qbar)/N

    # spatial resolution for integration, which is defined by the uncertainty
    # principle on the torus
    Nres = 3*int(np.ceil((4*np.pi*N)**0.5) + 1) # times 3 for good measure
    qs = np.linspace(*qlims, Nres)
    ps = np.linspace(*plims, Nres)
    qq, pp = np.meshgrid(qs, ps)

    # calculate Jacobi theta stuff
    thetas = np.zeros((N, Nres, Nres), dtype=complex)
    for m, qm in enumerate(qn):
        thetas[m] = jacobi_theta(qq, pp, N, qm)
    theta_sum = np.sum(np.abs(thetas)**2, axis=0)

    # evaluate function
    ff = f(qq, pp, *args)

    global calculate_element # avoid AttributeError

    def calculate_element(indices):
        n, m = indices
        out = ff*np.exp(1j*DPI*N*pp*(qn[n]-qn[m]))*thetas[n]*thetas[m].conj()
        out /= theta_sum
        out = simpson(simpson(out, x=qs), x=ps)
        return (n, m), out

    out = np.zeros((N, N), dtype=complex)
    nm = np.arange(0, N)
    offdiag = np.asarray(list(combinations(nm, 2)))
    diag = np.asarray(list(zip(nm, nm)))

    with Pool(None) as p:
        pmap_offdiag = p.map(calculate_element, offdiag)
        pmap_diag = p.map(calculate_element, diag)

    for (n, m), val in pmap_offdiag:
        out[n, m] = val
        out[m, n] = val.conjugate()

    for (n, _), val in pmap_diag:
        out[n, n] = val

    trace = np.real(np.linalg.trace(out))
    print(f'Trace: {trace}')
    if abs(1 - trace) > 1e-4:
        print('WARNING: trace is not 1. Integration limits may be wrong or point density too low.')
    return out


def coherent_ensemble_torus_traceless(f, N, qbar=QBAR, args=()):
    rho = coherent_ensemble_torus(f, N, qbar, args)
    return rho - np.eye(N)/N
