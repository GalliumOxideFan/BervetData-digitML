import numpy as np
from digitRecognition import whichNumber
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def testAll(UUT = np.load('TrainingData/UUT_trained.npy'), plotIncorrect = False, plotSize = (3,5)):
    """
    Tests the digit classification model on the full test dataset.

    The function computes the classification accuracy and optionally
    displays incorrectly classified digits.

    Parameters
    ----------
    UUT : ndarray, optional
        Projection matrices used for digit classification.

    plotIncorrect : bool, optional
        If True, displays incorrectly classified digits.

    plotSize : tuple, optional
        Grid size used when plotting incorrect classifications.

    Returns
    -------
    accuracy : float
        Classification accuracy on the test dataset.

    wronglyIdentified : ndarray
        Number of misclassified samples for each digit class.
    """

    # loads test set
    testDigits = np.load('HandwrittenDigits/TestDigits.npy')
    testLabels = np.load('HandwrittenDigits/TestLabels.npy').flatten()

    d = testDigits
    foundNumbers, res = whichNumber(d, UUT) # gets the classified digits by digitRecognition for all digits in the set
    incorrectNumbers = np.array(np.nonzero(foundNumbers-testLabels)).flatten() # gets the indexes of the numbers in the test set which were incorrectly classified
    wronglyIdentified = np.zeros(10) # Empty array to iterate over to calculate how many times each digit was classified wronly
    for number in incorrectNumbers: # check all wrong numbers
        wronglyIdentified[testLabels[number]]+=1 # add the correct number into the array
    
    accuracy = (testLabels.size-incorrectNumbers.size)/testLabels.size # calculate the accuracy of the classification
    print(f'Total numbers correctly classified: {testLabels.size-incorrectNumbers.size} \nTotal numbers incorrectly classified: {incorrectNumbers.size} \nTotal accuracy: {np.round(accuracy*100,2)}%')


    if plotIncorrect: # optional plotting of incorrectly classified digits
        state = {'page': 0} # semi global value for which page we are on
        plots = plotSize[0]*plotSize[1] # how many plots we have
        fig, ax = plt.subplots(plotSize[0], plotSize[1], squeeze=False, figsize=(14,12))

        def update_plot(page): # function to update plot when going to the next page and to instance the first page
            plt.suptitle(f'Incorrectly classified digits\nPage {page+1}', fontsize=30)
            for i in range(plots): # plot all images
                row = i//plotSize[1]
                col = i%plotSize[1]
                d = testDigits[:,incorrectNumbers[i+page*plots]] # get the image's vector
                D = np.reshape(d, (28, 28)).T # Reshaping a vector to a matrix
                ax[row, col].imshow(D, cmap ='gray') # Plot of the digit
                ax[row, col].axis('off')
                ax[row, col].set_title(f'Number classified as {foundNumbers[incorrectNumbers[i+page*plots]]}.\nCorrect is {testLabels[incorrectNumbers[i+page*plots]]}')
        update_plot(state['page']) # update plot for the first page
        plt.subplots_adjust(bottom=0.15) # give space for button
        button_ax = plt.axes([0.4, 0.05, 0.2, 0.075]) # add space for button
        button = Button(button_ax, 'Next Page') # create button

        def onClick(event): # what happens when clicking button
            state['page']+=1
            update_plot(state['page'])
            fig.canvas.draw_idle() # needed to update plot

        button.on_clicked(onClick) # call onClick when pressing button
        plt.show()
    return accuracy, wronglyIdentified
