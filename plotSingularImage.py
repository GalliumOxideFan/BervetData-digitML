import matplotlib.pyplot as plt
import numpy as np

U = np.load('TrainingData/U_trained.npy') # loads the singular vectors

plt.rcParams.update({   # Font size handling
        "font.size": 20,          # default for all text
    })

def plotDigit(i, numPlots = 3): # plot the digit i
    D = np.reshape(U[i,:,:numPlots], (28, 28, numPlots)).T # grabs the images for the numPlots first singular vectors
    fig, axs = plt.subplots(numPlots//5+1, min((numPlots,5)), squeeze=False, figsize=(12,5))
    plt.suptitle(f'The first {numPlots} singular images for digit {i}.')
    for j in range(numPlots): # plot the first singular images
        axs[j//5,j-j//5*5].imshow(D[j], cmap ='gray') # Plot of the digit
        axs[j//5,j-j//5*5].set_title(f'$u_{{{j+1}}}$')
        axs[j//5,j-j//5*5].axis('off')
    for j in range(axs.size-numPlots): # sets remaining plots axis to off to make them invisable, only important if numPlots > 5
        j = j+numPlots
        axs[j//5,j-j//5*5].axis('off')
    
    plt.savefig(f'SavedImages/singularImage{i}.pdf', bbox_inches="tight")
    plt.show()

plotDigit(3,3) # plots the first 3 singular values for the digit 3
plotDigit(8,3) # plots the first 3 singular values for the digit 8