import numpy as np
import matplotlib.pyplot as plt

S = np.load('TrainingData/S_trained.npy')


x = np.linspace(0, np.size(S[0]), np.size(S[0]))
plt.plot(x, S[0])
plt.show()