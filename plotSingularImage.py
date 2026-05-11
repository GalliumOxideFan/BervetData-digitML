import matplotlib.pyplot as plt
import numpy as np

U = np.load('TrainingData/U_trained.npy')

plt.rcParams.update({   # Font size handling
        "font.size": 20,          # default for all text
    })

def plotDigit(Ui, i, numPlots = 3):
    D = np.reshape(U[i,:,:numPlots], (28, 28, numPlots)).T
    fig, axs = plt.subplots(numPlots//5+1, min((numPlots,5)), squeeze=False, figsize=(12,5))
    plt.suptitle(f'Plotted singular images by the {numPlots} first columns in matrix U for digit {i}')
    for j in range(numPlots):
        axs[j//5,j-j//5*5].imshow(D[j], cmap ='gray') # Plot of the digit
        axs[j//5,j-j//5*5].set_title(f'$u_{{{j+1}}}$')
        axs[j//5,j-j//5*5].axis('off')
    for j in range(axs.size-numPlots):
        j = j+numPlots
        axs[j//5,j-j//5*5].axis('off')
    
    plt.savefig(f'SavedImages/singularImage{i}.pdf', bbox_inches="tight")
    plt.show()

plotDigit(U, 3,3)
plotDigit(U, 8,3)