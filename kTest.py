import numpy as np
import matplotlib.pyplot as plt
from testAllDigits import testAll
from training import UUT_update

ks = np.arange(8,26)

U = np.load('TrainingData/U_trained.npy')
accuracy = np.zeros(ks.size)
for k in ks:
    print('\nUpdating UUT...')
    UUTk = UUT_update(U,k)
    np.save('TrainingData/UUT_trained', UUTk)

    print(f'Testing images for k = {k}...')
    accuracy[k-ks[0]] = testAll()[0]


plt.figure(figsize=(12, 8))
plt.plot(ks, accuracy, 'g', linewidth = 3)
plt.xlabel('Number of basis functions, k', fontsize=22)
plt.ylabel('Computed accuracy', fontsize = 22)
plt.xticks(ks[::2], fontsize=17)
plt.yticks(fontsize=17)
plt.savefig("SavedImages/kTested.pdf", bbox_inches="tight")
plt.show()