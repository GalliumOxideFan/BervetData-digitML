import numpy as np
import matplotlib.pyplot as plt

S = np.load('TrainingData/S_trained.npy')


x = np.linspace(0, np.size(S[0]), np.size(S[0]))

plt.rcParams.update({   # Font size handling
        "font.size": 20,          # default for all text
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 10,
    })

plt.plot(x, S[3], label='Singular values for digit 3', linewidth=3)
plt.plot(x, S[8], label='Singular values for digit 8', linewidth=3)
plt.legend()
plt.xlabel('Singular value index')
plt.ylabel('Singular value')
plt.show()