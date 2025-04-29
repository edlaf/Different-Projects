import numpy as np
import numba

@numba.njit
def hayashi_yoshida_corr_numba(tA, vA, tB, vB):
    dA = np.diff(vA)
    dB = np.diff(vB)
    n = len(dA)
    m = len(dB)
    i = 0
    j = 0
    cov = 0.0
    var_A = 0.0
    var_B = 0.0

    while i < n and j < m:
        start_A = tA[i]
        end_A = tA[i+1]
        start_B = tB[j]
        end_B = tB[j+1]

        start_overlap = max(start_A, start_B)
        end_overlap = min(end_A, end_B)

        if start_overlap < end_overlap:
            cov += dA[i] * dB[j]

        if end_A <= end_B:
            i += 1
        else:
            j += 1

    var_A = np.sum(dA * dA)
    var_B = np.sum(dB * dB)

    corr = cov / np.sqrt(var_A * var_B)
    return corr 
