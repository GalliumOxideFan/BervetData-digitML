import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
def trainLM(k, numDigits=2000):
    TrainDigits = np.load('HandwrittenDigits/TrainDigits.npy')
    TrainLabels = np.load('HandwrittenDigits/TrainLabels.npy').flatten()


    idx = np.argsort(TrainLabels).flatten()
    sortedDigits = (TrainDigits.T)[idx].T
    A_matri = sortedDigits.T.reshape(10, -1, 784).transpose(0,2,1)

    print('Computing SVD...')
    U, S, Vt = np.linalg.svd(A_matri[:, :, :numDigits],compute_uv=True, full_matrices=True)

    print('SVD calculation complete... \nComputing U@Ut...')
    UUT = []
    for i in range(10):
        UUT.append(U[i,:,:k]@U[i,:,:k].T)
    
    print('Training complete!')
    return U, S, Vt, np.array(UUT)
