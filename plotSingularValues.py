import numpy as np
import matplotlib.pyplot as plt

S = np.load('TrainingData/S_trained.npy')

x = np.linspace(0, np.size(S[0]), np.size(S[0]))

plt.rcParams.update({   # Font size handling
        "font.size": 20,          # default for all text
    })

plt.plot(x, S[3], label='Singular values for digit 3', linewidth=3)
plt.legend()
plt.xlabel('Singular value index')
plt.ylabel('Singular value')
plt.show()


plt.plot(x, S[8], 'r', label='Singular values for digit 8', linewidth=3)
plt.legend()
plt.xlabel('Singular value index')
plt.ylabel('Singular value')
plt.show()