# MNIST Neural Network Classifier from Scratch

This project implements a simple feedforward neural network from scratch (without using deep learning libraries like TensorFlow or PyTorch) to classify handwritten digits from the MNIST dataset. The model is trained using mini-batch gradient descent and features one hidden layer with ReLU activation and a softmax output layer.

---

## Features

- Fully connected neural network with:
  - __Input layer__: 784 neurons (28x28 pixels)
  - __Hidden layer__: 128 neurons (`ReLU` activation)
  - __Output layer__: 10 neurons (`Softmax` activation)
- __One-hot__ encoded labels for classification
- Mini-batch gradient descent for efficient training
- `He` initialization for weight initialization
- Interactive prediction on test data

---

## Project Structure

```
.
├── mnist_classifier.py      # Main python script
└── MNIST dataset.npz        # Compressed NumPy file containing the MNIST dataset (`x_train`, `y_train`, `x_test`, `y_test`).
```

---

## Dependencies

To run this project, you will need the following Python libraries:
- `numpy` (for numerical operations)
- `matplotlib` (for plotting the results)

You can install them via `pip`:

```
pip install numpy matplotlib
```

---

## Dataset

The MNIST dataset should be stored in a `.npz` file named `MNIST dataset.npz`, containing:
- `x_train`: Training images (60000, 28, 28)
- `y_train`: Training labels (60000,)
- `x_test`: Test images (10000, 28, 28)
- `y_test`: Test labels (10000,)

---

## How to Run

1. Clone or download the repository.
2. Ensure you have the necessary dependencies installed (`numpy` and `matplotlib`).
3. Place the MNIST dataset file (`MNIST dataset.npz`) in the project directory.
4. Run the main script:

```
python mnist_classifier.py
```

### Interactive Prediction

After training, you can test the model by entering an index (0-9999) to view the model's prediction on test images.

---

## Model Training

The training process follows these steps:
- __Load and preprocess the dataset__: Normalize and reshape input data, convert labels to one-hot encoding.
- __Initialize weights and biases__: `He` initialization is used for better training stability.
- __Training loop__:
  - Shuffle the dataset.
  - Perform forward propagation.
  - Compute softmax activation and loss.
  - Perform backpropagation to update weights and biases.
  - Compute and display training accuracy.
- __Final evaluation__: Compute test accuracy on unseen data.

---

## Example Output (cp from my own device)

```
Epoch 1/25 - Train Acc: 70.45%
Epoch 2/25 - Train Acc: 88.72%
Epoch 3/25 - Train Acc: 89.17%
...
Epoch 25/25 - Train Acc: 95.12%

Final Test Accuracy: 94.98%

Enter indices 0-9999 to test predictions on test set:
Enter a number (0-9999) or 'q' to quit: 5
```

---

## Acknowledgments

- Inspired by basic neural network implementations in NumPy.
- MNIST dataset provided by Yann LeCun and colleagues.

---

__Enjoy real-time digit classification with this neural network! Contributions and feedback are welcome. 😊__


