import numpy as np
from numpy.linalg import svd

def svd_compress(mat: np.ndarray, ratio: float):
    '''
    Performs SVD to compress photo data (`mat`) to `ratio` of the original size.
    
    Returns three arrays of `np.ndarray` objects, indexed by the channels present
    in the original photo, corresponding to U, Sigma, V as they appear in the SVD
    formula.
    '''
    try:
        channels_cnt = mat.shape[2] # count the number of channels in the photo matrix
    except:
        channels_cnt = 1 # in case we get a grayscale image with no alpha, do this
        mat = mat[:,:,np.newaxis]
    L = [None] * channels_cnt
    for i in range(channels_cnt):
        L[i] = mat[:,:,i]
    m, n = L[0].shape[0], L[0].shape[1] # get dimensions of the photo
    # determine the number of singular values to keep to reach the desired
    # compression ratio.
    k = max(1, int(ratio * m * n / (m + n + 1)))
    U, Sigma, V_T, V = [[None] * channels_cnt for i in range(4)]
    U_C, Sigma_C, V_C = [[None] * channels_cnt for i in range(3)] # compressed matrices

    for i in range(channels_cnt):
        U[i], Sigma[i], V_T[i] = svd(L[i])
        V[i] = V_T[i].T
        U_C[i], Sigma_C[i], V_C[i] = U[i][:,:k], Sigma[i][:k], V[i][:, :k]

    return m, n, channels_cnt, U_C, Sigma_C, V_C
    # returns photo-specific information and the decomposition vectors