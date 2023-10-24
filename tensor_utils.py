import numpy as np
from cp_decompose import cp_tensor_power_method

canonical = np.eye(5)


def multilinear_map(T, V1, V2, V3):
    """
    Multi-linear map for a tensor T and matrices V1, V2, V3

    TODO can this function be made more efficient?
    """

    u = 0

    for i in range(T.shape[0]):
        for j in range(T.shape[0]):
            for l in range(T.shape[0]):
                u += T[i, j, l] * V1[i, :].reshape(V1.shape[1], 1, 1) * V2[j, :].reshape(1, V2.shape[1], 1) * V3[l, :].reshape(1, 1, V3.shape[1])

    return u.squeeze()


def multilinear_map_vec(T, v):
    """
    Special case of multi-linear map for a tensor T and a vector v (which replaces the three matrices)
    """

    u = T * tensor_prod(v, v, v)

    return u.sum()


def tensor_prod(v1, v2, v3):
    """
    Three-dimensional tensor product for vectors v1, v2, v3
    """

    return v1[:, np.newaxis, np.newaxis] * v2[np.newaxis, :, np.newaxis] * v3[np.newaxis, np.newaxis, :]


def whitening(M3, M2, n_models):
    """
    Applies the whitening operation to the third-moment tensor M3 given the second moment M2

    Returns the whitened tensor and the corresponding whitening matrix W
    """

    M2 += np.eye(M2.shape[0]) * 1e-5  # regularize M2 to make sure it is PSD
    w, v = np.linalg.eig(M2)

    # keep only top n_models eigenvalues/eigenvectors
    w = w[:n_models]
    v = v[:, :n_models]

    W = np.matmul(v, np.diag(w ** (-0.5)))  # whitening matrix

    return multilinear_map(M3, W, W, W), W


def power_iteration(T, v):
    """
    Applies one step of power iteration to vector v given tensor T
    """

    u = multilinear_map(T, np.eye(T.shape[0]), v[:, np.newaxis], v[:, np.newaxis])

    return u / np.linalg.norm(u)


def rtp(T, L, N, faster=True):
    """
    Runs RTP on tensor T.

    :param T: an orthogonally-decomposable tensor
    :param L: number of randomized restarts
    :param N: number of power iterazion updates
    :param faster: whether to use the faster algorithm in cp_decompose.py
    :return: the matrix of eigenvalues and the eigenvectors of T
    """

    k = T.shape[0]  # T assumed to be kxkxk

    if faster:
        T = T.reshape((k, k ** 2))
        evalues, evectors, _ = cp_tensor_power_method(T, k, L, N, None)
        return evalues, evectors

    evalues = []
    evectors = []

    for i in range(k):

        candidates = []

        # Generate several candidate vectors using random restarts
        for l in range(L):

            theta = np.random.normal(0, 1, k)
            theta = theta / np.linalg.norm(theta)

            # Apply N step of power iteration
            for n in range(N):
                # Apply power iteration
                theta = power_iteration(T, theta)

            candidates.append(theta)

        # Take best vector
        values = [multilinear_map_vec(T, c) for c in candidates]
        eigenvec = candidates[np.argmax(values)]

        # Apply N step of power iteration
        for n in range(N):
            eigenvec = power_iteration(T, eigenvec)

        eigenval = multilinear_map_vec(T, eigenvec)

        evalues.append(eigenval)
        evectors.append(eigenvec)

        # Deflate the tensor
        T = T - eigenval * tensor_prod(eigenvec, eigenvec, eigenvec)

    evalues = np.array(evalues)
    evectors = np.array(evectors).T

    return evalues, evectors