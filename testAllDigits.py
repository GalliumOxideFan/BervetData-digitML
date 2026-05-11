import numpy as np
from digitRecognition import whichNumber
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def testAll(plotIncorrect = False, plotSize = (3,5)):
    testDigits = np.load('HandwrittenDigits/TestDigits.npy')
    testLabels = np.load('HandwrittenDigits/TestLabels.npy').flatten()

    d = testDigits
    foundNumbers, res = whichNumber(d)
    incorrectNumbers = np.array(np.nonzero(foundNumbers-testLabels)).flatten()
    wronglyIdentified = np.zeros(10)
    for number in incorrectNumbers:
        wronglyIdentified[testLabels[number]]+=1
    
    accuracy = (testLabels.size-incorrectNumbers.size)/testLabels.size
    print(f'Total numbers correctly identified: {testLabels.size-incorrectNumbers.size} \nTotal numbers incorrectly identified: {incorrectNumbers.size} \nTotal accuracy: {np.round(accuracy*100,2)}%')


    if plotIncorrect:
        state = {'page': 0}
        plots = plotSize[0]*plotSize[1]
        fig, ax = plt.subplots(plotSize[0], plotSize[1], squeeze=False, figsize=(14,12))

        def update_plot(page):
            plt.suptitle(f'Incorrectly identified digits\nPage {page}', fontsize=30)
            for i in range(plots):
                row = i//plotSize[1]
                col = i%plotSize[1]
                d = testDigits[:,incorrectNumbers[i+page*plots]]
                D = np.reshape(d, (28, 28)).T # Reshaping a vector to a matrix
                ax[row,col].imshow(D, cmap ='gray') # Plot of the digit
                ax[row, col].axis('off')
                ax[row, col].set_title(f'Number identified as {foundNumbers[incorrectNumbers[i+page*plots]]}.\nCorrect is {testLabels[incorrectNumbers[i+page*plots]]}')
        update_plot(state['page'])
        plt.subplots_adjust(bottom=0.15)
        button_ax = plt.axes([0.4, 0.05, 0.2, 0.075])
        button = Button(button_ax, 'Next Page')

        def onClick(event):
            state['page']+=1
            update_plot(state['page'])
            fig.canvas.draw_idle()

        button.on_clicked(onClick)
        plt.show()
    return accuracy, wronglyIdentified
