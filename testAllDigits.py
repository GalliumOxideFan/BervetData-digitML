import numpy as np
from digitRecognition import whichNumber
import matplotlib.pyplot as plt

logs = False
plotIncorrect = True
plotSize = (5,3)

testDigits = np.load('HandwrittenDigits/TestDigits.npy')
testLabels = np.load('HandwrittenDigits/TestLabels.npy').flatten()

d = testDigits
foundNumbers = whichNumber(d)
a = np.where(foundNumbers==testLabels, foundNumbers, True)
incorrectNumbers = np.array(np.nonzero(foundNumbers-testLabels)).flatten()
if logs:
    for number in incorrectNumbers[0]:
        print(f'Incorrectly identified {testLabels[number]} as {foundNumbers[number]} at index {number}')
print(f'Total numbers correctly identified: {testLabels.size-incorrectNumbers.size} \nTotal numbers incorrectly identified: {incorrectNumbers.size} \nTotal accuracy: {np.round((testLabels.size-incorrectNumbers.size)/testLabels.size*100,2)}%')


if plotIncorrect:
    page = 2
    plots = plotSize[0]*plotSize[1]
    
    for i in range(plots):
        plt.subplot(plotSize[0], plotSize[1], i+1)
        d = testDigits[:,incorrectNumbers[i+page*plots]]
        D = np.reshape(d, (28, 28)).T # Reshaping a vector to a matrix
        plt.imshow(D, cmap ='gray') # Plot of the digit
        plt.title(f'Number identified as {foundNumbers[incorrectNumbers[i+page*plots]]}.\nCorrect is {testLabels[incorrectNumbers[i+page*plots]]}')
    plt.show()
