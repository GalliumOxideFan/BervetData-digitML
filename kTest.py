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
    accuracy[k-ks[0]] = testAll()

plt.plot(ks, accuracy, 'g')
plt.xlabel('Number of basis functions, k', fontsize=20)
plt.ylabel('Computed accuracy', fontsize = 20)
plt.show()