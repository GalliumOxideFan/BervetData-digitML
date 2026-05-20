import numpy as np
import training

# params
k=15 # number of singular vectors used.
numDigits = 400 # number of training digits used per digit (0-9), max 24000, max reasonable 10000 ish

U, S, Vt, UUTk = training.trainLM(k, numDigits)

np.save('TrainingData/U_trained', U)
np.save('TrainingData/S_trained', S)
np.save('TrainingData/Vt_trained', Vt)
np.save('TrainingData/UUT_trained', UUTk)