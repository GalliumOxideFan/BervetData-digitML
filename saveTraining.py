import numpy as np
import training

U, S, Vt = training.trainLM()

np.save('TrainingData/U_trained', U)
np.save('TrainingData/S_trained', S)
np.save('TrainingData/Vt_trained', Vt)