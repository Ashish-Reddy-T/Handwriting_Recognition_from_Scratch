import numpy as np
import matplotlib.pyplot as plt

# Load and process the MNIST data!
def get_mnist():
    with np.load('MNIST dataset.npz') as f:
        images = f["x_train"] # Shape --> (60000, 28, 28) (60000 ---> number of images, 28*28 -> the pixel dimensions)
        labels = f["y_train"] # Shape --> (60000,)
    images = images.astype("float32") / 255 # Normalize images to be in the range [0,1]
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2])) # Reshaping --> (60000, 784(28*28))
    labels = np.eye(10)[labels] # Creates a one-hot encoded matrix
    """
    For example, if the first label is 3, it becomes `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`. 
    This makes sense for neural networks because each output node can represent one class.
    This reshapes it to (60000, 10)
    """
    return images, labels

# Function for introducing non-linearity
def softmax(x):
    exp_x = np.exp(x - np.max(x)) # Subtract max(x) for stability
    return exp_x / np.sum(exp_x)

hidden_neurons = 64

# Weights !!
## w_i_h = np.random.uniform(-0.5, 0.5, (4, 784))  # 4 hidden nodes, 784 input nodes
## w_h_o = np.random.uniform(-0.5, 0.5, (10, 4))   # 10 output nodes (for digits 0-9), 4 hidden nodes
w_i_h = np.random.uniform(-0.5, 0.5, (hidden_neurons, 784))  # 64 hidden nodes, 784 input nodes
w_h_o = np.random.uniform(-0.5, 0.5, (10, hidden_neurons))   # 10 output nodes, 64 hidden nodes

# Biases !!
## b_i_h = np.zeros((4, 1))   # Bias for hidden layer
## b_h_o = np.zeros((10, 1))  # Bias for output layer (10 nodes for digits 0-9)
b_i_h = np.zeros((hidden_neurons, 1))   # Bias for hidden layer
b_h_o = np.zeros((10, 1))  # Bias for output layer

# Load Data
images, labels = get_mnist()

nr_correct = 0
learn_rate = 0.01
# Training Loop
epochs = 3
for epoch in range(epochs):
    for img, l in zip(images, labels):
        # Reshape input and labels
        img = img.reshape(784, 1)  # Reshape to column vector
        l = l.reshape(10, 1)    # Reshape to column vector for 10 classes

        # Forward propagation
        h_pre = b_i_h + w_i_h @ img
        h = 1 / (1 + np.exp(-h_pre)) # Sigmoid activation in hidden layer
        
        o_pre = b_h_o + w_h_o @ h
        ## o = 1 / (1 + np.exp(-o_pre))
        o = softmax(o_pre)

        ## # Cost calculation (remove indexing with [0])  [Mean Squared Error]
        ## e = 1 / len(o) * np.sum((o - l)**2)

        e = -np.sum(l*np.log(o + 1e-8)) # Cross Entropy Loss for better accuracy as compared with MSE

        nr_correct += int(np.argmax(o) == np.argmax(l)) # Calculating accuracy

        # Backpropagation (Output --> Hidden)
        delta_o = o - l
        w_h_o += -learn_rate * (delta_o @ np.transpose(h))
        b_h_o += -learn_rate * delta_o

        # Backpropagation (Hidden --> Input)
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
        w_i_h += -learn_rate * (delta_h @ np.transpose(img))
        b_i_h += -learn_rate * delta_h

    print(f"Epoch {epoch+1}: Accuracy = {nr_correct / len(images) * 100:.2f}%")
    nr_correct = 0  # Reset for next epoch  

print("\nTraining complete! Enter numbers between 0-59999 to see predictions.")
print("Enter 'q' to quit the program.\n")

while True:
    try:
        user_input = input("Enter a number (0-59999) or 'q' to quit: ")
        
        if user_input.lower() == 'q':
            print("\nExiting program...")
            break
            
        index = int(user_input)
        if index < 0 or index >= 60000:
            print("Please enter a valid number between 0 and 59999")
            continue
            
        img = images[index]
        plt.figure(figsize=(6, 6))  # Create a new figure with specific size
        plt.imshow(img.reshape(28,28), cmap="Greys")

        h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
        h = 1 / (1 + np.exp(-h_pre)) # Hidden Layer with Sigmoid
        o_pre = b_h_o + w_h_o @ h
        ## o = 1 / (1 + np.exp(-o_pre)) # Sigmoid Function for non-linearity
        o = softmax(o_pre)

        plt.title(f"Predicted digit: {np.argmax(o)}")
        plt.draw()  # Draw the plot
        plt.show(block=True)
        
    except ValueError:
        print("Please enter a valid number or 'q' to quit")
    except KeyboardInterrupt:
        print("\nExiting program...")
        break




