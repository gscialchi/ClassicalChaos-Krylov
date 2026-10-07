from itertools import product

import numpy as np



def get_lyapunov(jacobian, map, q0, p0, n_steps, n_transient=10000):
    """
    Get Lyapunov exponent with QR factorization.
    """
    x, y = q0, p0
    for _ in range(n_transient):
        x, y = map(x, y)

    Q = np.eye(2)
    lyap = 0
    for _ in range(n_steps):
        J = jacobian(x, y)
        Z = J @ Q #
        Q, R = np.linalg.qr(Z)
        lyap += np.max(np.log(np.abs(np.diag(R))))
        x, y = map(x, y)
    return lyap/n_steps


def get_average_lyapunov(jacobian, map, Nqp, n_steps=1000):
    """
    Phase-space averaged Lyapunov exponent on a grid of Nqp points.
    """
    qs = np.linspace(*[0, 1], int(Nqp**0.5)+1)
    ps = np.linspace(*[0, 1], int(Nqp**0.5)+1)

    out = 0
    for (q, p) in product(qs, ps):
        out += get_lyapunov(jacobian, map, q, p, n_steps)
    return out/Nqp
