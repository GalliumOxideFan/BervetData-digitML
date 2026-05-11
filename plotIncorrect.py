import numpy as np
from training import UUT_update
from testAllDigits import testAll
import matplotlib.pyplot as plt

# params
k = 15 # number of singular vectors used.

U = np.load('TrainingData/U_trained.npy') # load saved U matrix
UUT_update(U, k) # make sure we are using the correct UUT for set k

acc, wrong = testAll(plotIncorrect=True) # tests all digits in the test set against the SVD and plots the ones that was incorrectly classified

# makes a plot of how often each digit was misclassified
x = np.arange(10)
plt.figure(figsize=(12, 8))
plt.bar(x, wrong)
plt.title(f'Misclassifications for each digit with k = {k}', fontsize=26)
plt.xlabel('Digit', fontsize =22)
plt.ylabel('Number of misclassifications', fontsize=22)
plt.xticks(x, fontsize=16)
plt.yticks(fontsize=16)
plt.savefig(f'SavedImages/wrongIdentified.pdf', bbox_inches="tight")
plt.show()