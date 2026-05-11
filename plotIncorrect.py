import numpy as np
from training import UUT_update
from testAllDigits import testAll
import matplotlib.pyplot as plt

k = 15
U = np.load('TrainingData/U_trained.npy')
UUT_update(U, k)

acc, wrong = testAll(plotIncorrect=True)

x = np.arange(10)
plt.figure(figsize=(12, 8))
plt.bar(x, wrong)
plt.title('Wrong estimations for each digit', fontsize=26)
plt.xlabel('Digit', fontsize =22)
plt.ylabel('Number of times gotten wrong', fontsize=22)
plt.xticks(x, fontsize=16)
plt.yticks(fontsize=16)
plt.savefig(f'SavedImages/wrongIdentified.pdf', bbox_inches="tight")
plt.show()