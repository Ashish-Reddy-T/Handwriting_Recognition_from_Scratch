# MNIST Neural Network Classifier

This project implements a simple neural network to classify handwritten digits from the MNIST dataset. The neural network uses the following structure:
- __Input layer__: 784 neurons (28x28 pixels per image flattened into a vector)
- __Hidden layer__: 128 neurons with ReLU activation
- __Output layer__: 10 neurons (representing digits 0 to 9) with softmax activation for classification

The model is trained using stochastic gradient descent with mini-batches, and the cross-entropy loss function is used for training.

---

## Project Structure

```
.
├── mnist_classifier.py      # Python script implementing the neural network, training loop, and interactive prediction.
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

The MNIST dataset contains 60,000 images of handwritten digits for training and 10,000 images for testing. The images are 28x28 grayscale pixels. The labels correspond to the digits (0-9) each image represents.

---

## How It Works

- __Data Loading__: The `get_mnist()` function loads the MNIST dataset from a .npz file and processes it by normalizing pixel values to the range [0, 1] and reshaping them into vectors (784,).
- __Neural Network__:
  - The network consists of:
    - A __hidden layer__ with ReLU activation.
    - An __output layer__ with softmax activation.
  - __Forward Propagation__: The input image is passed through the network layers to compute the predicted output.
  - __Loss Function__: The model uses cross-entropy loss for better classification performance compared to mean squared error.
  - __Backpropagation__: The weights are updated using stochastic gradient descent with mini-batches.
- Training:
  - The model is trained for 25 epochs, with mini-batches of size 128.
  - The training accuracy is printed at the end of each epoch.
- Testing:
  - After training, the model is evaluated on the test set to calculate its final accuracy.
- Interactive Predictions:
  - The user can enter an index between 0-9999 to visualize a test image and see the predicted label alongside the true label.

---

## How to Run

1. Clone or download the repository.
2. Ensure you have the necessary dependencies installed (`numpy` and `matplotlib`).
3. Place the MNIST dataset file (`MNIST dataset.npz`) in the project directory.
4. Run the main script:

```
python mnist_classifier.py
```
5. During training, the model will output the accuracy for each epoch.
6. After training is complete, you can input a test image index (between 0-9999) to see predictions.

---

## Example Output

```
Epoch 1/25 - Train Acc: 92.45%
Epoch 2/25 - Train Acc: 93.72%
...
Final Test Accuracy: 94.53%

Enter indices 0-9999 to test predictions on test set:
Enter a number (0-9999) or 'q' to quit: 5
```

## Hyperparameters

- __Hidden Neurons__: 128
- __Batch Size__: 128
- __Learning Rate__: 0.01
- __Epochs__: 25

---

__Enjoy real-time digit classification with this neural network! Contributions and feedback are welcome. 😊__


