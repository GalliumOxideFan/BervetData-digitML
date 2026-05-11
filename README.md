# Handwritten Digit Recognition using SVD

This project implements a handwritten digit classifier based on Singular Value Decomposition (SVD) and low-rank subspace approximation. Each digit class (0–9) is represented by a low-dimensional linear subspace, and classification is performed by minimizing projection residuals.

---


## Usage

### Downloading project

```bash
git clone https://github.com/GalliumOxideFan/BervetData-digitML
cd BervetData-digitML
```

---

### Save trained matrices

```bash
python saveTraining.py
```

---

### Plot incorrectly classified numbers

```bash
python plotIncorrect.py
```

---

### Plot accuracy for different ammount of singular vectors

```bash
python kTest.py
```

### Plot accuracy for different ammount of training images

```bash
python numTrainingImpact.py
```

### Plot singular values

```bash
python plotSingularValue.py
```

### Plot singular images
```bash
python plotSingularImage.py
```
### Notes
- Make sure the directories: HandwrittenDigits/, SavedImages/ and TrainingData exists.
- The data for training the model needs to be placed in HandwrittenDigits/. Extract the archive "HandwrittenDigits.rar" here.
- The speed of testing the test set seems to be wildly inconsistent across diffrent hardware.

### Requirements

- Python 3
- NumPy
- Matplotlib

Install dependencies with:

```bash
pip install numpy matplotlib
```

---
## Method

For each digit class:

1. Training images are collected into a matrix
2. Singular Value Decomposition is computed

   $$
   A = U\Sigma V^T
   $$

3. The first \(k\) singular vectors are used to define a low-dimensional subspace
4. A projection matrix is constructed

   $$
   P = U_k U_k^T
   $$

To classify a test digit \(d\):

1. Project the digit onto every digit subspace
2. Compute reconstruction residuals

   $$
   r_i = ||d - P_i d||
   $$

3. Select the digit class with the smallest residual

---

## Results

The classifier was tested on a dataset of 40,000 handwritten digits.

- Best accuracy achieved: approximately 94.9%
- Optimal range:

  $$
  k \approx 19 \text{ to } 20
  $$

- Digits such as 0 and 1 were easiest to classify
- Digits such as 5, 8, and 9 produced more misclassifications due to similar shapes

The results show that handwritten digits can be well approximated using low-dimensional linear subspaces.

---

## Features

- SVD-based handwritten digit classification
- Projection-based nearest-subspace classifier
- Visualization of singular values
- Visualization of singular images
- Accuracy evaluation for different values of \(k\)
- Misclassification analysis
- Interactive viewer for incorrectly classified digits

---

## Project Structure

### `training.py`

Contains:

- `trainLM(k, numDigits)`
  - Computes SVD for each digit class
  - Constructs projection matrices

- `UUT_update(U, k)`
  - Recomputes projection matrices for a new value of \(k\)

---

### `digitRecognition.py`

Contains:

- `whichNumber(d, UUT)`
  - Classifies digit vectors using reconstruction residuals

---

### `testAllDigits.py`

Contains:

- `testAll(...)`
  - Evaluates classifier on the full test set
  - Computes accuracy
  - Optionally visualizes incorrectly classified digits

---

### Plotting Scripts

- `plotSingularValues.py`
  - Plots singular values for selected digits

- `plotSingularImage.py`
  - Plots singular images obtained from the left singular vectors

- `plotIncorrect.py`
  - Displays incorrectly classified digits
  - Plots misclassification counts per digit

- `kTest.py`
  - Evaluates classification accuracy for different values of \(k\)

- `numTrainingImpact.py`
  - Evaluates how the number of training images affects accuracy

---


## Dataset Format

The project expects the following files:

```text
HandwrittenDigits/
├── TrainDigits.npy
├── TrainLabels.npy
├── TestDigits.npy
└── TestLabels.npy
```

Each digit image is represented as a flattened 28×28 grayscale image:

$$
28 \times 28 = 784
$$

---

## Example Accuracy Test

The effect of varying the number of singular vectors was tested for

$$
k = 8,9,\dots,25
$$

The accuracy increased from approximately 93.0% at \(k=8\) to approximately 94.9% near \(k=20\), after which the performance stabilized slightly below the peak.

---

## Notes

- The classifier is entirely linear algebra based
- No neural networks are used
- Performance depends strongly on the choice of \(k\)
- The method is fast once the SVDs have been computed
- Similar-looking digits can produce overlapping subspaces
- This ReadMe has been co-written by AI

---


