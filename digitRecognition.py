import numpy as np

def whichNumber(d, UUT = np.load('TrainingData/UUT_trained.npy')):
    """
    Classifies digit vectors by projecting them onto each digit subspace
    and selecting the class with the smallest reconstruction error.

    Parameters
    ----------
    d : ndarray
        Input digit vector(s) to classify.

    UUT : ndarray, optional
        Projection matrices for each digit class.

    Returns
    -------
    identifiedDigits : ndarray
        Predicted digit labels.

    res : ndarray
        Reconstruction residuals for each digit class.
    """
    Ud = UUT @ d
    res = np.linalg.norm(d - Ud, axis=1) # calculates the residual for d when projected onto the subspace of U with k number of singular vectors
    identifiedDigits = np.argmin(res, axis=0) # gets the digit associated to the smallest residual and decides that to be the classified digit
    return identifiedDigits, res