import numpy as np
from digitRecognition import whichNumber

logs = True

testDigits = np.load('HandwrittenDigits/TestDigits.npy')
testLabels = np.load('HandwrittenDigits/TestLabels.npy').flatten()

correctNumbers = 0
incorrectNumbers = 0
d = testDigits
foundNumbers = whichNumber(d)
for i in range(testLabels.size):
    d = testDigits[:,i]
    foundNumber = whichNumber(d)
    if logs:
        if foundNumber == testLabels[i]:
            print(f'Correctrly identified number {i+1} as {foundNumber}')
            correctNumbers+=1
        else:
            print(f'Could not identify number {i+1} as {testLabels[i]}, identified number was {foundNumber}')
            incorrectNumbers += 1

print(f'Total numbers correctly identified: {correctNumbers} \nTotal numbers incorrectly identified: {incorrectNumbers} \nTotal accuracy: {np.round(correctNumbers/(correctNumbers+incorrectNumbers)*100,2)}')