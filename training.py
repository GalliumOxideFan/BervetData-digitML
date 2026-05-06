import numpy as np
import matplotlib.pyplot as plt
def trainLM():
    TrainDigits = np.load('HandwrittenDigits/TrainDigits.npy')
    TrainLabels = np.load('HandwrittenDigits/TrainLabels.npy').flatten()


    idx = np.argsort(TrainLabels).flatten()
    sortedDigits = (TrainDigits.T)[idx].T
    A_matri = sortedDigits.T.reshape(10, -1, 784)

    U, S, Vt = np.linalg.svd(A_matri[:, :400, :], full_matrices=False)
    return U, S, Vt