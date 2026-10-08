import numpy as np

from cc_krylov.maps import QBAR, PBAR


def pos_to_chord(op, qbar=QBAR, pbar=PBAR):
    """
    Calculate chord function of an operator from its position representation.

    See: Notebook C (yellow, Hornocal), p.89
    TODO: write notes.
    """
    N = op.shape[0]
    out = np.zeros_like(op)
    nunu, mumu = np.meshgrid(np.arange(N), np.arange(N))

    i, j = np.tril_indices_from(op)
    Z = np.zeros_like(op)
    Z[i-j, j] = op[i, j]
    Z = np.fft.fft(Z, axis=1)
    out += np.exp(-1j*2*(np.pi/N)*mumu*nunu)*Z

    i, j = np.triu_indices_from(op, k=1)
    Z = np.zeros_like(op)
    Z[i-j, i] = op[i, j]
    Z = np.fft.fft(Z, axis=1)

    out += np.exp(1j*2*np.pi*pbar)*Z
    out *= np.exp(1j*(np.pi/N)*mumu*nunu - 1j*2*(np.pi/N)*qbar*nunu)/N**0.5
    return out


def chord_to_pos(op, qbar=QBAR, pbar=PBAR):
    """
    Calculate position representation of an operator from its chord function.

    See: Notebook C (yellow, Hornocal), pages 89-90
    TODO: write notes.
    """
    N = op.shape[0]
    out = np.zeros_like(op)

    nunu, mumu = np.meshgrid(np.arange(N), np.arange(N))
    phase = np.exp(1j*np.pi/N * (mumu + 2*qbar) * nunu)
    ifft = np.fft.ifft(op*phase, axis=1)

    m, n = np.tril_indices_from(ifft)
    out[m, n] += ifft[m-n, n]

    m, n = np.triu_indices_from(ifft, k=1)
    out[m, n] += np.exp(-1j*2*np.pi*pbar)*ifft[m-n+N, n]
    out *= N**0.5
    return out


def chord_second_moment(op, qbar=QBAR, pbar=PBAR):
    """
    Calculate second moment of the operator's Chord function.
    """
    N = op.shape[0]
    op_chord = pos_to_chord(op, qbar, pbar)
    w = np.abs(op_chord)**2
    w = w/np.sum(w) # normalize operator in Hilbert-Schmidt norm

    n = np.arange(N)/N
    n_ = np.mod(n+1/2, 1) - 1/2
    xx, yy = np.meshgrid(n_, n_)
    xi2 = xx**2 + yy**2
    out = np.sum((w*xi2).flatten())**0.5
    return out
