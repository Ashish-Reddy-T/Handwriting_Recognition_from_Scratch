import numpy as np
import matplotlib.pyplot as plt

# Load and process the MNIST data!
def get_mnist():
    with np.load('MNIST dataset.npz') as f:
        x_train = f["x_train"]  # Shape --> (60000, 28, 28) (60000 ---> number of images, 28*28 -> the pixel dimensions)
        y_train = f["y_train"]  # Shape --> (60000,)
        x_test = f["x_test"]    # Shape --> (10000, 28, 28)
        y_test = f["y_test"]    # Shape --> (10000,)
    
    # Process training data
    x_train = x_train.astype("float32") / 255  # Normalize images to be in the range [0,1]
    x_train = x_train.reshape(-1, 28*28)       # (60000, 784)
    y_train = np.eye(10)[y_train]              # (10000, 10)

    # Process test data
    x_test = x_test.astype("float32") / 255
    x_test = x_test.reshape(-1, 28*28)    # (10000, 784)
    y_test = np.eye(10)[y_test]           # (10000, 10)

    """
    For example, if the first label is 3, it becomes `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`. 
    This makes sense for neural networks because each output node can represent one class.
    This reshapes it to (60000, 10)
    """

    return x_train, y_train, x_test, y_test

    ## images = images.astype("float32") / 255 # Normalize images to be in the range [0,1]
    ## images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2])) # Reshaping --> (60000, 784(28*28))
    ## labels = np.eye(10)[labels] # Creates a one-hot encoded matrix
    ## """
    ## For example, if the first label is 3, it becomes `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`. 
    ## This makes sense for neural networks because each output node can represent one class.
    ## This reshapes it to (60000, 10)
    ## """
    ## return images, labels

# Function for introducing non-linearity
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis = 0, keepdims=True)) # Subtract max(x) for stability
    return exp_x / np.sum(exp_x, axis = 0, keepdims=True)

# All the hyperparameteres assembled
hidden_neurons = 128
batch_size = 128
learn_rate = 0.01
epochs = 10

# Improved initialization (He initialization for ReLU)
w_i_h = np.random.randn(hidden_neurons, 784) * np.sqrt(2./784)
w_h_o = np.random.randn(10, hidden_neurons) * np.sqrt(2./hidden_neurons)
b_i_h = np.zeros((hidden_neurons, 1))
b_h_o = np.zeros((10, 1))

# Load all data
x_train, y_train, x_test, y_test = get_mnist()

# Training Loop with mini-batches (for faster processing)
for epoch in range(epochs):
    # Shuffle training data
    permutation = np.random.permutation(x_train.shape[0])
    x_train = x_train[permutation]
    y_train = y_train[permutation]

    total_correct = 0

    for i in range(0, x_train.shape[0], batch_size):

        # Get batch
        x_batch = x_train[i:i+batch_size].T  # (784, batch_size)
        y_batch = y_train[i:i+batch_size].T  # (10, batch_size)

        # Forward pass
        h_pre = w_i_h @ x_batch + b_i_h
        h = np.maximum(0, h_pre)  # ReLU
        o_pre = w_h_o @ h + b_h_o
        o = softmax(o_pre)

        # Calculate accuracy
        predictions = np.argmax(o, axis=0)
        true_labels = np.argmax(y_batch, axis=0)
        total_correct += np.sum(predictions == true_labels)

         # Backward pass
        delta_o = o - y_batch
        delta_h = (w_h_o.T @ delta_o) * (h_pre > 0)  # ReLU derivative

        # Update parameters (average over batch)
        w_h_o -= learn_rate * (delta_o @ h.T) / batch_size
        b_h_o -= learn_rate * np.mean(delta_o, axis=1, keepdims=True)
        w_i_h -= learn_rate * (delta_h @ x_batch.T) / batch_size
        b_i_h -= learn_rate * np.mean(delta_h, axis=1, keepdims=True)

        # # # Reshape input and labels
        # # img = img.reshape(784, 1)  # Reshape to column vector
        # # l = l.reshape(10, 1)    # Reshape to column vector for 10 classes

        # # # Forward propagation
        # # h_pre = b_i_h + w_i_h @ img
        # # h = 1 / (1 + np.exp(-h_pre)) # Sigmoid activation in hidden layer
        
        # # o_pre = b_h_o + w_h_o @ h
        # # ## o = 1 / (1 + np.exp(-o_pre))
        # # o = softmax(o_pre)

        # # ## # Cost calculation (remove indexing with [0])  [Mean Squared Error]
        # # ## e = 1 / len(o) * np.sum((o - l)**2)

        # # e = -np.sum(l*np.log(o + 1e-8)) # Cross Entropy Loss for better accuracy as compared with MSE

        # # nr_correct += int(np.argmax(o) == np.argmax(l)) # Calculating accuracy

        # # # Backpropagation (Output --> Hidden)
        # # delta_o = o - l
        # # w_h_o += -learn_rate * (delta_o @ np.transpose(h))
        # # b_h_o += -learn_rate * delta_o

        # # # Backpropagation (Hidden --> Input)
        # # delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
        # # w_i_h += -learn_rate * (delta_h @ np.transpose(img))
        # # b_i_h += -learn_rate * delta_h
    
     # Calculate epoch statistics
    train_acc = total_correct / x_train.shape[0] * 100
    print(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.2f}%")

    # # print(f"Epoch {epoch+1}: Accuracy = {nr_correct / len(images) * 100:.2f}%")
    # # nr_correct = 0  # Reset for next epoch  

# Final test evaluation
h_pre = w_i_h @ x_test.T + b_i_h
h = np.maximum(0, h_pre)
o = softmax(w_h_o @ h + b_h_o)
test_acc = np.mean(np.argmax(o, axis=0) == np.argmax(y_test.T, axis=0)) * 100
print(f"\nFinal Test Accuracy: {test_acc:.2f}%")

# Interactive prediction (now using test set)
print("\nEnter indices 0-9999 to test predictions on test set:")

while True:
    try:
        user_input = input("Enter a number (0-9999) or 'q' to quit: ")
        if user_input.lower() == 'q':
            break
        
        index = int(user_input)

        if 0 <= index < 10000:
            img = x_test[index].reshape(28, 28)
            plt.imshow(img, cmap="Greys")
            
            # Model prediction
            h_pre = w_i_h @ x_test[index] + b_i_h
            h = np.maximum(0, h_pre)
            o = softmax(w_h_o @ h + b_h_o)
            
            plt.title(f"True: {np.argmax(y_test[index])}, Pred: {np.argmax(o)}")
            plt.show()
        else:
            print("Please enter a number between 0-9999")
            
    except ValueError:
        print("Invalid input. Please enter a number or 'q' to quit.")




