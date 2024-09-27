
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tqdm import trange

import os


path_to_main = os.path.dirname(os.path.realpath(__file__))

class Dataset:
    def __init__(self, ns, sig, range, seed = None):

        self.ns = ns  
        self.sig = sig  
        self.range = range 
        self.index = np.arange(2 * ns) 

        # Create the numpy.random.Generator instance with an optional seed for reproducibility
        self.rng = np.random.default_rng(seed)

        # Generate random angles (theta) and radii for the spirals
        self.theta_1 = self.rng.uniform(0, 2 * np.pi * range[1], size=ns)
        self.theta_2 = self.rng.uniform(0, 2 * np.pi * range[1], size=ns)

        
        self.radius_spiral_1 = self.range[0] + (self.theta_1 * (self.range[1] - self.range[0])) / (2 * np.pi)
        self.x_1 = self.radius_spiral_1 * np.cos(self.theta_1 + np.pi)
        self.y_1 = self.radius_spiral_1 * np.sin(self.theta_1 + np.pi)

        self.radius_spiral_2 = self.range[0] + (self.theta_2 * (self.range[1] - self.range[0])) / (2 * np.pi)
        self.x_2 = self.radius_spiral_2 * np.cos(self.theta_2)
        self.y_2 = self.radius_spiral_2 * np.sin(self.theta_2)

        # Add Gaussian white noise to the data
        self.x_1 += self.rng.normal(0, sig, ns)
        self.y_1 += self.rng.normal(0, sig, ns)
        self.x_2 += self.rng.normal(0, sig, ns)
        self.y_2 += self.rng.normal(0, sig, ns)

        self.data_1 = [self.x_1, self.y_1]
        self.data_2 = [self.x_2, self.y_2]

        # Create the input dataset in matrix form
        self.input_data = np.concatenate([self.data_1, self.data_2], axis=1).T

        # Create the binary classification labels for the output data
        self.output_data = np.concatenate(([0] * ns, [1] * ns))

    def plot(self):

        plt.scatter(self.x_1, self.y_1, label="Class 1", color='red')
        plt.scatter(self.x_2, self.y_2, label="Class 2", color='blue')
        plt.legend()
        plt.title("Generated Spiral Data")
        plt.axis("equal")  # Ensures equal scaling on both axes
        plt.savefig(f"{path_to_main}/generated_data_plt.pdf")

    def get_batch(self, batch_size):
        # Choose a random batch size sample
        choices = self.rng.choice(self.index, size=batch_size)
        return self.input_data[choices], self.output_data[choices]


    
class LinearLayers(tf.Module):
    def __init__(self, num_out, activation, name = None):
        super().__init__(name = name)
        self.num_out = num_out
        self.activation = activation
        self.__is_built = False
        print(f"Activation function: {self.activation}")

    def build(self, num_in, index = 0):
        # Initialize weights and biases
        self.weight = tf.Variable(
            tf.random.normal([num_in, self.num_out]) * 0.01, name = 'w' + str(index)
        )
        self.bias = tf.Variable(
            tf.zeros([1, self.num_out]), name = 'b' + str(index)
        )

        self.__is_built = True

    def __call__(self, x):
        if not self.__is_built:
            raise ValueError("Linear Layer not built")
        
        # Pass Wx + b through the nonlinearity:
        return self.activation((x @ self.weight) + self.bias)
        
# Multi-layer Perceptron implemented using linear layers
class MLP(tf.Module):
    def __init__(self, 
                num_inputs, 
                num_outputs, 
                num_hidden_layers, 
                hidden_layer_width, 
                hidden_activation = tf.nn.leaky_relu, 
                output_activation = tf.identity, 
                name = None):
        
        super().__init__(name=name)

        # Save the activation functions
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.layers = []

        # Pass the input layer through the first hidden layer
        self.layers.append(LinearLayers(num_out=hidden_layer_width, activation = self.hidden_activation))

        # Add the hidden layers 
        for i in range(num_hidden_layers):
            self.layers.append(LinearLayers(num_out=hidden_layer_width, activation = self.hidden_activation))

        self.layers.append(LinearLayers(num_out=num_outputs, activation = self.output_activation))

    def build(self, input_dimension):
        num_in = input_dimension

        # Assign input/output relationship for each layer
        for i, layer in enumerate(self.layers):
            layer.build(num_in, index = i)
            num_in = layer.num_out
    
    def __call__(self, x):
        # Forward pass through the MLP
        value = x
        for layer in self.layers:
            value = layer(value)
        return value
    

class Adam(tf.Module):
    ''' Adam algorithm referenced on the tensorflow website: 
    https://www.tensorflow.org/guide/core/optimizers_core
    '''
    def __init__(self, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, ep=1e-7):
      # Initialize the Adam parameters
      self.beta_1 = beta_1
      self.beta_2 = beta_2
      self.learning_rate = learning_rate
      self.ep = ep
      self.t = 1.
      self.v_dvar, self.s_dvar = [], []
      self.title = f"Adam: learning rate={self.learning_rate}"
      self.built = False

    def apply_gradients(self, grads, vars):
      # Set up moment and RMSprop slots for each variable on the first call
      if not self.built:
        for var in vars:
          v = tf.Variable(tf.zeros(shape=var.shape))
          s = tf.Variable(tf.zeros(shape=var.shape))
          self.v_dvar.append(v)
          self.s_dvar.append(s)
        self.built = True
      # Perform Adam updates
      for i, (d_var, var) in enumerate(zip(grads, vars)):
        # Moment calculation
        self.v_dvar[i] = self.beta_1*self.v_dvar[i] + (1-self.beta_1)*d_var
        # RMSprop calculation
        self.s_dvar[i] = self.beta_2*self.s_dvar[i] + (1-self.beta_2)*tf.square(d_var)
        # Bias correction
        v_dvar_bc = self.v_dvar[i]/(1-(self.beta_1**self.t))
        s_dvar_bc = self.s_dvar[i]/(1-(self.beta_2**self.t))
        # Update model variables
        var.assign_sub(self.learning_rate*(v_dvar_bc/(tf.sqrt(s_dvar_bc) + self.ep)))
      # Increment the iteration counter
      self.t += 1.


class BinaryCrossEntropyLoss:
    def __init__(self, epsilon = 1e-7):
        # Initialize BCE with a small epsilon to prevent numerical instability
        self.epsilon = epsilon

    def sigmoid(self, logits):
        # Create sigmoid function needed to compute the sigmoid of the logits
        return 1/(1 + tf.exp(-logits))
    
    def __call__(self, logits, labels):
        labels = tf.cast(labels, tf.float32)
        predictions = self.sigmoid(logits)
        # Clip the predictions
        predictions = tf.clip_by_value(predictions, self.epsilon, 1. - self.epsilon)
        # Calculate loss
        loss = -(labels * tf.math.log(predictions) + (1 - labels) * tf.math.log(1 - predictions))
        return tf.reduce_mean(loss)


from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.base import BaseEstimator, ClassifierMixin


class MLPWrapper(BaseEstimator, ClassifierMixin):
    '''
    Create custom wrapper so that we can still use the scikit-learn estimator.
        - Make the model behave like a scikit-learn estimator
        - Incorporate predict, fit methods necessary
    '''
    def __init__(self, mlp):
        self.mlp = mlp
        self.is_fitted_ = True
        self.classes_ = np.array([0, 1])    # Define class labels

    def predict_proba(self, X):
        logits = self.mlp(X).numpy()
        probs = tf.sigmoid(logits).numpy()
        # Return the probabilities in a 2 column array (b/c of binary classification)
        return np.c_[1 - probs, probs]  # Apparently sklearn expects [P(class 0), P(class 1)]
    
    def fit(self, X, y):
        self.is_fitted_ = True
        self.classes_ = np.unique(y)    # Automatically infer the classes from the data
        return self
    

def plot_decision_boundary_with_sklearn(mlp, dataset):
    # Wrap the MLP model using the custom wrapper class above
    mlp_wrapper = MLPWrapper(mlp)

    X = dataset.input_data
    y = dataset.output_data

    DecisionBoundaryDisplay.from_estimator(
        estimator=mlp_wrapper,
        X = X,
        response_method = "predict_proba",
        grid_resolution = 100,              # Resolution of grid
        plot_method = 'contourf',
        alpha = 0.6,                        # Decision boundary transparency
        cmap = "RdBu"
    )

    # Include data points in the plot, overlayed
    plt.scatter(dataset.x_1, dataset.y_1, color='red', label='Class 1')
    plt.scatter(dataset.x_2, dataset.y_2, color='blue', label="Class 2")
    plt.legend()
    plt.title("MLP Decision Boundary")
    plt.savefig(f"{path_to_main}/MLP_Decision_Boundary_plt.pdf")

def accuracy(logits, labels):
    predicted = tf.round(tf.sigmoid(logits))
    labels = tf.cast(labels, tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype = tf.float32))


'''

BEGIN MAIN CODE SEGMENT AND TRAINING


'''

# Generate spiral data:
data = Dataset(500, 0.01, (0.75, 5))
data.plot()

# Initialize BCE loss:
bce_loss = BinaryCrossEntropyLoss()

# Initialize MLP and Adam optimizer:
mlp = MLP(num_inputs=2, num_outputs=1, num_hidden_layers=3, hidden_layer_width=64)
mlp.build(2)

learning_rate = 0.001       # Initialize Adam with default learning rate
adam_optimizer = Adam(learning_rate)
epochs = 10000
batch_size = 64

'''
TRAINING LOOP
'''

for epoch in trange(epochs):
    x_batch, y_batch = data.get_batch(batch_size)
    y_batch = tf.expand_dims(y_batch, axis=-1)

    with tf.GradientTape() as tape:
        logits = mlp(x_batch)
        loss = bce_loss(logits, y_batch)

    gradients = tape.gradient(loss, mlp.trainable_variables)

    adam_optimizer.apply_gradients(gradients, mlp.trainable_variables)

    if epoch % 1000 == 0:
        acc = accuracy(logits, y_batch)
        print(f"Epoch {epoch}: Loss = {loss.numpy()}, Accuracy = {acc.numpy()}\n")

'''
Create plot of decision boundary
'''
plot_decision_boundary_with_sklearn(mlp, data)





    