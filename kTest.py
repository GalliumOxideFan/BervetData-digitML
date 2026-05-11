import numpy as np
import matplotlib.pyplot as plt
from testAllDigits import testAll
from training import UUT_update

ks = np.arange(8,26) # which k:s we should test

U = np.load('TrainingData/U_trained.npy') # load singular matrix
accuracy = np.zeros(ks.size) # empty array for accuracy for each k in ks
for k in ks: # gets the accuracy for each value of k in ks
    print('\nUpdating UUT...')
    UUTk = UUT_update(U,k)
    print(f'Testing images for k = {k}...')
    accuracy[k-ks[0]] = testAll(UUTk)[0]


# plots the accuracy for the different k:s
plt.figure(figsize=(12, 8))
plt.plot(ks, accuracy, 'g', linewidth = 3)
plt.xlabel('Number of basis functions, k', fontsize=22)
plt.ylabel('Computed accuracy', fontsize = 22)
plt.xticks(ks[::2], fontsize=17)
plt.yticks(fontsize=17)
plt.savefig("SavedImages/kTested.pdf", bbox_inches="tight")
plt.show()