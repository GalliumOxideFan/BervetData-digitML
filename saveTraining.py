import numpy as np
import training

k=15
numDigits = 400
U, S, Vt, UUTk = training.trainLM(k, numDigits)

np.save('TrainingData/U_trained', U)
np.save('TrainingData/S_trained', S)
np.save('TrainingData/Vt_trained', Vt)
np.save('TrainingData/UUT_trained', UUTk)
