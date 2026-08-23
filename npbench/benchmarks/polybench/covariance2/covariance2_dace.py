import numpy as np
import dace as dc

M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))


@dc.program
def kernel(float_n: dc.float64, data: dc.float64[N, M]):

    # np.cov(np.transpose(data)): rows of the transpose are the M variables and its
    # columns the N observations, so the estimator is centered per column of data and
    # normalised by N - 1 (ddof=1). np.cov does not touch its input, so unlike the
    # covariance kernel this centers into a separate buffer.
    mean = np.mean(data, axis=0)
    centered = data - mean
    cov = np.zeros((M, M), dtype=data.dtype)
    for i in range(M):
        cov[i, i:M] = centered[:, i] @ centered[:, i:M] / (float_n - 1.0)
        cov[i:M, i] = cov[i, i:M]

    return cov
