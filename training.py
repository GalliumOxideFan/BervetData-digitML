import numpy as np

def trainLM(k, numDigits=400): 
    """
    Trains a low-rank handwritten digit model using SVD.

    The training images are grouped by digit class (0-9), and an SVD is
    computed for each class using the first `numDigits` samples. The first
    `k` singular vectors are used to construct projection matrices
    U_k U_k^T for later classification or reconstruction.

    Parameters
    ----------
    k : int
        Number of singular vectors used in the low-rank approximation.

    numDigits : int, optional
        Number of training images per digit class. Default is 400.

    Returns
    -------
    U, S, Vt : ndarray
        SVD components for each digit class.

    UUT : ndarray
        Projection matrices U_k U_k^T for each digit class.
    """

    # import data for training
    TrainDigits = np.load('HandwrittenDigits/TrainDigits.npy')
    TrainLabels = np.load('HandwrittenDigits/TrainLabels.npy').flatten() # needs to be flattened to be vector

    # sort the data according to digits, we get an A matrix for each digit
    idx = np.argsort(TrainLabels).flatten()
    sortedDigits = (TrainDigits.T)[idx].T
    A_matri = sortedDigits.T.reshape(10, -1, 784).transpose(0,2,1)

    print('Computing SVD...')
    U, S, Vt = np.linalg.svd(A_matri[:, :, :numDigits],compute_uv=True, full_matrices=True) # calculates SVD using numpy

    print(f'SVD calculation complete... \nComputing U@Ut with k = {k}...')
    UUT = []
    for i in range(10): # precalculates U@Ut for each digit
        UUT.append(U[i,:,:k]@U[i,:,:k].T)
    
    print('Training complete!')
    return U, S, Vt, np.array(UUT)
def UUT_update(U, k): 
    """
    Computes the projection matrices U_k @ U_k^T using the first
    `k` singular vectors for each digit class.

    Parameters
    ----------
    U : ndarray
        Left singular vectors.

    k : int
        Number of singular vectors used.

    Returns
    -------
    ndarray
        Projection matrices for each digit class, UUT.
    """
    UUT = []
    for i in range(10):
        UUT.append(U[i,:,:k]@U[i,:,:k].T)
    return np.array(UUT)