from .utils import *

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size,
                 activation_hidden='relu', loss_fn='mse'):

        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.num_layers = len(self.layer_sizes) - 1
        
        # initialize weights
        self.weights = [np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * 0.01
                        for i in range(self.num_layers)]
        self.biases = [np.zeros((1, self.layer_sizes[i+1])) for i in range(self.num_layers)]
        
        # get activation & derivative from dict
        act_func, act_deriv = activations[activation_hidden]
        self.activations = [act_func]*(self.num_layers-1) + [activations['linear'][0]]  # output always linear
        self.activations_deriv = [act_deriv]*(self.num_layers-1) + [activations['linear'][1]]

        # get loss & derivative from dict
        self.loss_fn, self.loss_derivative = losses[loss_fn]
        
        # ADD: Initialize loss history storage
        self.train_loss_history = []
        self.test_loss_history = []

    # === forward pass === #
    def forward(self, X):
        
        self.a_values = [X]
        
        for i in range(self.num_layers):
            z = np.dot(self.a_values[i], self.weights[i]) + self.biases[i]
            a = self.activations[i](z)
            self.a_values.append(a)
            
        return self.a_values[-1]
    
    # === backward pass === #
    def backward(self, X, y_true):
        
        m = y_true.shape[0]
        self.dW = [None]*self.num_layers
        self.db = [None]*self.num_layers
        
        dA = self.loss_derivative(y_true, self.a_values[-1])
        
        for i in reversed(range(self.num_layers)):
            dZ = dA * self.activations_deriv[i](self.a_values[i+1])
            self.dW[i] = np.dot(self.a_values[i].T, dZ) / m
            self.db[i] = np.sum(dZ, axis=0, keepdims=True) / m
            dA = np.dot(dZ, self.weights[i].T)
    
    # === update weights === #
    def update_weights(self, lr=0.01):
        for i in range(self.num_layers):
            self.weights[i] -= lr * self.dW[i]
            self.biases[i] -= lr * self.db[i]
    
    # === training loop === #
    def train(self, X, y, epochs=100, lr=0.01, mode='batch', batch_size=32, 
              X_test=None, y_test=None, log_interval=None):

        n_samples = X.shape[0]
        self.train_loss_history = []
        self.test_loss_history = []

        for epoch in range(epochs):
            
            # === training step === #
            if mode == 'batch':
                y_pred = self.forward(X)
                loss = self.loss_fn(y, y_pred)
                self.backward(X, y)
                self.update_weights(lr)
                train_loss = loss
            
            elif mode == 'stochastic':
                loss_epoch = 0
                for i in range(n_samples):
                    xi, yi = X[i:i+1], y[i:i+1]
                    y_pred = self.forward(xi)
                    loss = self.loss_fn(yi, y_pred)
                    self.backward(xi, yi)
                    self.update_weights(lr)
                    loss_epoch += loss
                train_loss = loss_epoch / n_samples
            
            elif mode == 'minibatch':
                perm = np.random.permutation(n_samples)
                X_shuffled, y_shuffled = X[perm], y[perm]
                loss_epoch = 0
                num_batches = 0
                for start in range(0, n_samples, batch_size):
                    end = start + batch_size
                    X_batch, y_batch = X_shuffled[start:end], y_shuffled[start:end]
                    y_pred = self.forward(X_batch)
                    loss = self.loss_fn(y_batch, y_pred)
                    self.backward(X_batch, y_batch)
                    self.update_weights(lr)
                    loss_epoch += loss
                    num_batches += 1
                train_loss = loss_epoch / num_batches
            
            else:
                raise ValueError("Mode must be 'batch', 'stochastic', or 'minibatch'")
            
            self.train_loss_history.append(train_loss)
            
            # === compute test loss === #
            if X_test is not None and y_test is not None:
                y_test_pred = self.predict(X_test)
                test_loss = self.loss_fn(y_test, y_test_pred)
                self.test_loss_history.append(test_loss)
            else:
                test_loss = None
            
            # === logging === #
            if log_interval is not None:
                if (epoch % log_interval == 0) or (epoch == epochs - 1):
                    if test_loss is not None:
                        print(f"Epoch {epoch+1}/{epochs} || Train Loss: {train_loss:.6f} || Test Loss: {test_loss:.6f}")
                    else:
                        print(f"Epoch {epoch+1}/{epochs} || Train Loss: {train_loss:.6f}")

        return self.train_loss_history
    
    # === predict === #
    def predict(self, X):
        return self.forward(X)