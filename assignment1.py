import numpy as np
import struct
#加载数据
def load_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, 784)
    return images

def load_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

train_images = load_images(r"C:\Users\23969\desktop\train-images-idx3-ubyte")
train_labels = load_labels(r"C:\Users\23969\desktop\train-labels-idx1-ubyte")
test_images = load_images(r"C:\Users\23969\desktop\t10k-images-idx3-ubyte")
test_labels = load_labels(r"C:\Users\23969\desktop\t10k-labels-idx1-ubyte")

#构建三层神经网络
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights1 = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size)
        self.weights2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))
        self.activation = activation

    def relu(self, z):
        return np.maximum(0, z)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def forward(self, x):
        self.z1 = np.dot(x, self.weights1) + self.bias1
        if self.activation == 'relu':
            self.a1 = self.relu(self.z1)
        else:
            raise ValueError("Unsupported activation function")
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def compute_loss(self, y_true, y_pred, reg_lambda=0.0):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true])
        loss = np.sum(log_likelihood) / m
        l2_penalty = reg_lambda * (np.sum(self.weights1**2) + np.sum(self.weights2**2))
        return loss + l2_penalty

    def backprop(self, x, y_true, reg_lambda=0.0):
        m = x.shape[0]
        y_true_one_hot = np.eye(self.output_size)[y_true]
        delta3 = self.a2 - y_true_one_hot
        dw2 = (1 / m) * np.dot(self.a1.T, delta3) + (reg_lambda * self.weights2)
        db2 = (1 / m) * np.sum(delta3, axis=0, keepdims=True)

        delta2 = np.dot(delta3, self.weights2.T) * self.relu_derivative(self.z1)
        dw1 = (1 / m) * np.dot(x.T, delta2) + (reg_lambda * self.weights1)
        db1 = (1 / m) * np.sum(delta2, axis=0)

        return dw1, db1, dw2, db2

#训练网络
def train_network(network, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, learning_rate=0.01, lr_decay=0.99, reg_lambda=0.01):
    num_train = X_train.shape[0]
    num_batches = num_train // batch_size
    train_loss_history = []
    val_loss_history = []
    val_acc_history = []
    best_val_acc = 0
    best_weights = (network.weights1, network.weights2, network.bias1, network.bias2)
    
    for epoch in range(epochs):
        for i in range(num_batches):
            batch_indices = np.random.choice(num_train, batch_size)
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            # Forward pass
            output = network.forward(X_batch)

            # Compute loss
            loss = network.compute_loss(y_batch, output, reg_lambda)
            train_loss_history.append(loss)

            # Backpropagation
            dw1, db1, dw2, db2 = network.backprop(X_batch, y_batch, reg_lambda)

            # Update weights and biases
            network.weights1 -= learning_rate * dw1
            network.bias1 -= learning_rate * db1
            network.weights2 -= learning_rate * dw2
            network.bias2 -= learning_rate * db2

        # Validate the model
        val_output = network.forward(X_val)
        val_loss = network.compute_loss(y_val, val_output, reg_lambda)
        val_loss_history.append(val_loss)

        # Calculate validation accuracy
        val_predictions = np.argmax(val_output, axis=1)
        val_accuracy = np.mean(val_predictions == y_val)
        val_acc_history.append(val_accuracy)

        # Save the model if it's the best so far
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_weights = (network.weights1.copy(), network.weights2.copy(), network.bias1.copy(), network.bias2.copy())
            print(f"New best model saved at epoch {epoch} with validation accuracy {val_accuracy}")

        # Learning rate decay
        learning_rate *= lr_decay

        print(f"Epoch {epoch}, Loss: {loss}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

    # Load the best model weights
    network.weights1, network.weights2, network.bias1, network.bias2 = best_weights
    return train_loss_history, val_loss_history, val_acc_history

# Example usage
nn = NeuralNetwork(784, 128, 10)
train_loss, val_loss, val_acc = train_network(nn, train_images, train_labels, test_images, test_labels, epochs=50, batch_size=64, learning_rate=0.01, lr_decay=0.95, reg_lambda=0.001)

#可视化训练和验证过程
import matplotlib.pyplot as plt

def plot_history(train_loss, val_loss, val_acc):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_acc)
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

plot_history(train_loss, val_loss, val_acc)
