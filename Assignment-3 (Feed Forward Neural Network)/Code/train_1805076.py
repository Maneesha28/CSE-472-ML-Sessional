import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix
import pickle
import torchvision.datasets as ds
from torchvision import transforms
import pandas as pd

np.random.seed(42)

def data_processing(dataset):
    processed_dataset = []

    for data in dataset:
        image_value = np.array(list(map(float, data[0].numpy().flatten())))
        label = np.array(data[1])

        if not np.isnan(label):
            processed_dataset.append((image_value, label))

    processed_dataset = np.array(processed_dataset, dtype=object)

    np.random.shuffle(processed_dataset)

    return processed_dataset

def split_features_labels(dataset):
    features = np.array([data[0] for data in dataset])
    labels = np.array([data[1] for data in dataset])
    labels = labels - 1  # convert labels to 0-25
    return features, labels

# split the dataset into training and validation 85/15
def split_dataset(dataset, split=0.85):
    split = int(len(dataset) * split)
    train_data = dataset[:split]
    validation_data = dataset[split:]

    return train_data, validation_data

train_validation_dataset = ds.EMNIST(root='./data', split='letters',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)


independent_test_dataset = ds.EMNIST(root='./data', split='letters',
                             train=False,
                             transform=transforms.ToTensor())

processed_train_data = data_processing(train_validation_dataset)
processed_test_data = data_processing(independent_test_dataset)

train_data, validation_data = split_dataset(processed_train_data)
x_train, y_train = split_features_labels(train_data)
x_validation, y_validation = split_features_labels(validation_data)
x_test, y_test = split_features_labels(processed_test_data)

# y should have int type
y_train = y_train.astype(int)
y_validation = y_validation.astype(int)
y_test = y_test.astype(int)

x_train.shape

"""**All our matrices are of the form \[: , batch_size\]**"""

def one_hot_encoding(labels, num_classes):
    # convert labels into 0-25
    # labels = labels - 1
    # create one-hot encoding
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

# we have to use cross-entropy loss for multi-class classification
def cross_entropy_loss(y, y_hat):
    # clip output to avoid log(0) error
    y_hat = np.clip(y_hat, 1e-8, None)
    return -np.sum(y * np.log(y_hat))

def cross_entropy_loss_prime(y, y_hat):
    return y_hat-y

class AdamOptimizer:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 0

    def update(self, gradient, learning_rate):
        self.t += 1

        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        return learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO: update parameters and return input gradient
        pass

class Dense(Layer):
    def __init__(self, input_size, output_size):
        # He initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.bias = np.zeros(output_size)
        self.weights_optimizer = AdamOptimizer()
        self.bias_optimizer = AdamOptimizer()


    def forward(self, input):
        self.input = input
        return np.dot(self.input, self.weights) + self.bias

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.dot(output_gradient, self.weights.T) / output_gradient.shape[1]
        weights_gradient = np.dot(self.input.T, output_gradient) / output_gradient.shape[1]
        bias_gradient = np.mean(output_gradient, axis=0)

        self.weights = self.weights - self.weights_optimizer.update(weights_gradient, learning_rate)
        self.bias = self.bias - self.bias_optimizer.update(bias_gradient, learning_rate)

        return input_gradient

class Activation(Layer):

    def forward(self, input):
        pass

    def backward(self, output_gradient, learning_rate):
        pass

class ReLU(Activation):

    def relu(self, x):
        return np.maximum(0, x)

    def relu_prime(self, x):
        return (x > 0).astype(float)

    def forward(self, input):
        self.input = input
        return self.relu(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.relu_prime(self.input))

class Softmax(Layer):
    def forward(self, input):
        input = input.astype(float)
        exp = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = exp / np.sum(exp, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient, learning_rate):
        grad = output_gradient/output_gradient.shape[-1]
        return grad

class DropoutLayer(Layer):
    def __init__(self, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.mask = None  # Mask to store which units are dropped out during training

    def forward(self, input, training=True):
        self.input = input  # Store input for backward pass
        if training:
            # During training, create a binary mask where units are dropped out with probability dropout_rate
            self.mask = (np.random.rand(*input.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            self.output = input * self.mask
        else:
            # During testing or inference, scale the output by (1 - dropout_rate)
            self.output = input * (1 - self.dropout_rate)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # The backward pass is simply the element-wise multiplication with the stored mask
        return output_gradient * self.mask

class FNN:
    def __init__(self, layers, name):
        self.name = name
        self.layers = layers

    def forward(self, x, training=True):
        for layer in self.layers:
            if isinstance(layer, DropoutLayer):
                x = layer.forward(x, training=training)
            else:
                x = layer.forward(x)
        return x


    def backward(self, y_true, y_pred, learning_rate):
        grad_output = cross_entropy_loss_prime(y_true, y_pred)
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, learning_rate)

    def mini_batch_train(self, x_train, y_train, batch_size, learning_rate):

        total_loss = 0.0
        correct_predictions = 0
        total_trained_samples = 0

        for i in range(0, len(x_train), batch_size):
                  x_batch = x_train[i:i+batch_size]
                  y_batch = y_train[i:i+batch_size]

                  # Forward pass
                  y_pred = self.forward(x_batch)

                  one_hot_y_batch = one_hot_encoding(y_batch, 26)

                  # Backward pass
                  self.backward(one_hot_y_batch, y_pred, learning_rate)

                  batch_loss = cross_entropy_loss(one_hot_y_batch, y_pred)
                  total_loss += batch_loss.sum()

                  # Calculate training accuracy
                  correct_predictions += np.sum(np.argmax(y_pred, axis=1) == np.array(y_batch))
                  total_trained_samples += len(y_batch)

        train_mean_loss = total_loss / len(x_train)
        train_accuracy = (correct_predictions / total_trained_samples) * 100

        return train_accuracy, train_mean_loss


    def train_and_validate(self, x_train, y_train, x_val, y_val, epochs, batch_size, learning_rate):
        all_train_losses = []
        all_val_losses = []
        all_train_acc = []
        all_val_acc = []
        all_train_f1_scores = []
        all_val_f1_scores = []
        best_val_f1_score = 0.0

        for _ in tqdm(range(epochs)):

            train_accuracy, train_mean_loss = self.mini_batch_train(x_train, y_train, batch_size, learning_rate)
            y_pred = self.predict(x_val)
            val_f1_score, val_accuracy, val_mean_loss = self.performance_metrices(y_val, y_pred)

            all_train_losses.append(train_mean_loss)
            all_val_losses.append(val_mean_loss)
            all_train_acc.append(train_accuracy)
            all_val_acc.append(val_accuracy)
            all_val_f1_scores.append(val_f1_score)
            # print(f'Epoch {epoch + 1}, Loss: {train_mean_loss:.4f}, Training Accuracy: {train_accuracy:.4f}%')
            # print(f'Validation Loss: {val_mean_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}%, Validation F1 Score: {val_f1_score:.4f}')


            if(val_f1_score > best_val_f1_score):
                best_val_f1_score = val_f1_score
                best_accuracy = val_accuracy
                best_loss = val_mean_loss
                best_predicted_labels = np.argmax(y_pred, axis=1)

        generate_graph(all_train_losses, all_val_losses, epochs, 'Loss', self.name)
        generate_graph(all_train_acc, all_val_acc, epochs, 'Accuracy', self.name)
        generate_graph(None, all_val_f1_scores, epochs, 'F1_Score', self.name)
        plot_confusion_matrix(y_val, best_predicted_labels, 26, self.name)

        return best_val_f1_score, best_accuracy, best_loss

    def predict(self, x_val):
      y_pred = self.forward(x_val, training=False)
      return y_pred

    def performance_metrices(self, y_true, y_pred):
        correct_predictions = 0
        total_val_samples = 0

        one_hot_y = one_hot_encoding(y_true, 26)

        total_loss = np.sum(cross_entropy_loss(one_hot_y, y_pred))
        mean_loss = total_loss/len(y_true)

        # Calculate validation accuracy
        predicted_labels = np.argmax(y_pred, axis=1)
        correct_predictions += np.sum(predicted_labels == np.array(y_true))
        total_val_samples += len(y_true)
        val_accuracy = (correct_predictions / total_val_samples) * 100

        # Calculate f1_score
        val_f1_score = f1_score(y_true, predicted_labels, average='macro')

        return val_f1_score, val_accuracy, mean_loss

def generate_graph(train_data, val_data, epochs, y_title, model_name):
    # Create a DataFrame for Seaborn plotting
    df = pd.DataFrame({
        'Epoch': range(1, epochs + 1),
        'Training': train_data,
        'Validation': val_data
    })

    # Set Seaborn style
    sns.set_theme(style="white")

    # Plot the data
    plt.figure(figsize=(10, 6))
    if(y_title != 'F1_Score'):
      ax = sns.lineplot(x='Epoch', y='Training', data=df, label='Training')
    ax = sns.lineplot(x='Epoch', y='Validation', data=df, label='Validation')

    # Set plot labels and title
    plt.title(y_title)
    plt.xlabel('Epoch')
    plt.ylabel('value')

    # Show legend
    plt.legend()

    # Show the plot
    fig_name = model_name + '_' + y_title
    plt.savefig(fig_name)


def plot_confusion_matrix(y_true, y_pred, classes, model_name):

    cm = confusion_matrix(y_true, y_pred)
    # Plotting the confusion matrix using seaborn
    plt.figure(figsize=(classes, classes))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    fig_name = model_name + '_Confusion_Matrix'
    plt.savefig(fig_name)

"""# Models"""

# fnn1 = FNN([
#     Dense(input_size=28*28, output_size=128),
#     ReLU(),
#     DropoutLayer(dropout_rate=0.5),
#     Dense(input_size=128, output_size=26),  # Assuming 26 classes for letters
#     Softmax()
# ], 'Model_1')

# # Train the FNN
# fnn1.train_and_validate(x_train, y_train, x_validation, y_validation, epochs=100, batch_size=1024, learning_rate=0.005)

# # main function
# fnn2 = FNN([
#     Dense(input_size=28*28, output_size=256),
#     ReLU(),
#     DropoutLayer(dropout_rate=0.3),
#     Dense(input_size=256, output_size=128),
#     ReLU(),
#     DropoutLayer(dropout_rate=0.5),
#     Dense(input_size=128, output_size=26),  # Assuming 26 classes for letters
#     Softmax()
# ], 'Model_2')

# # Train the FNN
# fnn2_f1_score = fnn2.train_and_validate(x_train, y_train, x_validation, y_validation, epochs=50, batch_size=1024, learning_rate=0.005)

# # main function
# fnn3 = FNN([
#     Dense(input_size=28*28, output_size=128),
#     ReLU(),
#     Dense(input_size=128, output_size=64),
#     ReLU(),
#     DropoutLayer(dropout_rate=0.4),
#     Dense(input_size=64, output_size=32),
#     ReLU(),
#     DropoutLayer(dropout_rate=0.3),
#     Dense(input_size=32, output_size=26),  # Assuming 26 classes for letters
#     Softmax()
# ], 'Model_3')

# # Train the FNN
# fnn3.train_and_validate(x_train, y_train, x_validation, y_validation, epochs=50, batch_size=1024, learning_rate=0.005)

# with open('best_model.pickle', 'wb') as file:
#     pickle.dump(fnn1, file)

# with open('best_model1.pickle', 'rb') as file:
#     loaded_model = pickle.load(file)

# y_pred = loaded_model.predict(x_test)

# loaded_model.performance_metrices(y_test, y_pred)