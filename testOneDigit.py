import numpy as np
from digitRecognition import whichNumber
import matplotlib.pyplot as plt

index = 0
plot = True

testDigits = np.load('HandwrittenDigits/TestDigits.npy')
testLabels = np.load('HandwrittenDigits/TestLabels.npy').flatten()

d = testDigits[:,index]
foundNumber, res = whichNumber(d)
print(f'Found number was {foundNumber}')

if plot:
    D = np.reshape(d, (28, 28)).T # Reshaping a vector to a matrix
    plt.imshow(D, cmap ='gray') # Plot of the digit
    plt.axis('off')
    plt.title(f'Identified as {foundNumber}')
    plt.show()