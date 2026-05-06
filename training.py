import numpy as np
import matplotlib.pyplot as plt
<<<<<<< HEAD
<<<<<<< HEAD
from scipy.linalg import lstsq
def trainLM(k, numDigits=2400):
=======
def trainLM():
>>>>>>> 9da8df90f52ae8fc25d03ea707de3ec4e29d4067
=======
def trainLM():
>>>>>>> 9da8df90f52ae8fc25d03ea707de3ec4e29d4067
    TrainDigits = np.load('HandwrittenDigits/TrainDigits.npy')
    TrainLabels = np.load('HandwrittenDigits/TrainLabels.npy').flatten()


    idx = np.argsort(TrainLabels).flatten()
    sortedDigits = (TrainDigits.T)[idx].T
<<<<<<< HEAD
<<<<<<< HEAD
    A_matri = sortedDigits.T.reshape(10, -1, 784).transpose(0,2,1)

    U, S, Vt = np.linalg.svd(A_matri[:, :, :numDigits],compute_uv=True, full_matrices=True)

    UUT = []
    for i in range(10):
        UUT.append(U[i,:,:k]@U[i,:,:k].T)
    
    return U, S, Vt, np.array(UUT)
=======
    A_matri = sortedDigits.T.reshape(10, -1, 784)

    U, S, Vt = np.linalg.svd(A_matri[:, :400, :], full_matrices=False)
    return U, S, Vt
>>>>>>> 9da8df90f52ae8fc25d03ea707de3ec4e29d4067
=======
    A_matri = sortedDigits.T.reshape(10, -1, 784)

    U, S, Vt = np.linalg.svd(A_matri[:, :400, :], full_matrices=False)
    return U, S, Vt
>>>>>>> 9da8df90f52ae8fc25d03ea707de3ec4e29d4067
