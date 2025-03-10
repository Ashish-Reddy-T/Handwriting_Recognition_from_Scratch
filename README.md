# MNIST Neural Network Classifier

This project implements a simple neural network to classify handwritten digits from the MNIST dataset. The neural network uses the following structure:
- __Input layer__: 784 neurons (28x28 pixels per image flattened into a vector)
- __Hidden layer__: 128 neurons with ReLU activation
- __Output layer__: 10 neurons (representing digits 0 to 9) with softmax activation for classification

The model is trained using stochastic gradient descent with mini-batches, and the cross-entropy loss function is used for training.

---

## Project Structure

- `mnist_classifier.py`
