import numpy as np
import matplotlib.pyplot as plt

S = np.load('TrainingData/S_trained.npy')

x = np.linspace(0, np.size(S[0]), np.size(S[0]))

plt.rcParams.update({   # Font size handling
        "font.size": 20,          # default for all text
    })

def plot(digit):
    plt.figure(figsize=(12, 8))
    plt.plot(x, S[digit], label=f'Singular values for digit {digit}', linewidth=3, color=(0,1-digit/10,digit/10))
    plt.legend()
    plt.xlabel('Singular value index')
    plt.ylabel('Singular value')
    plt.savefig(f'SavedImages/singularValue{digit}.pdf', bbox_inches="tight")
    plt.show()

plot(3)
plot(8)