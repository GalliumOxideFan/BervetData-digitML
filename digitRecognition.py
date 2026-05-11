import numpy as np

def whichNumber(d):

    UUT = np.load('TrainingData/UUT_trained.npy')
    Ud = UUT @ d
    res = np.linalg.norm(d - Ud, axis=1)
    minRes = np.min(res, axis=0)

    identifiedDigits = np.argmin(res, axis=0)
    return identifiedDigits, res