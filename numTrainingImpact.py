import numpy as np
import matplotlib.pyplot as plt
from training import trainLM
from testAllDigits import testAll

#params
k=15 # number of singular vectors used.
numDigits = np.array((10, 50, 100, 200, 400, 1000, 2000)) # which number of digits to test and plot

accuracy = np.zeros(numDigits.size) # empty array 

for i in range(numDigits.size): # test accuracy for each number of digits
    dig = numDigits[i]
    print(f'\nTraining for {dig} digits...')
    U, S, Vt, UUTk = trainLM(k, dig) # calculates U, S, Vt, UUTk for the number of digits with testAllDigits
    print(f'\nTesting images for {dig} digits...')
    accuracy[i] = testAll(UUTk)[0] # adds the calculated accuracy to an array to plot
    print('\n--------------------------------------')

positions = np.arange(len(numDigits)) # needed for equal width bars
plt.figure(figsize=(12, 8))
plt.bar(positions, accuracy)
plt.xticks(positions, numDigits, fontsize=17)
plt.xlabel('Images used for training per digit', fontsize=22)
plt.ylabel('Computed accuracy with $k=15$', fontsize = 22)
plt.ylim((np.min(accuracy)*0.9, 1))
plt.yticks(fontsize=17)
plt.savefig("SavedImages/numDigits.pdf", bbox_inches="tight")
plt.show()