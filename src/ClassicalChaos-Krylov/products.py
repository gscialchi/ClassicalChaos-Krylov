import numpy as np


def frobenius_prod(op1, op2):
    return np.trace(op1.conj().T @ op2)


def operator_prod(op1, op2, hbar=1):
    return frobenius_prod(op1, op2)/(2*np.pi*hbar)


def operator_norm(op, hbar=1):
    return np.sqrt(np.real(operator_prod(op, op, hbar)))
