from utils import *

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, activation_hidden=relu, activation_hidden_derivative=relu_derivative):
        
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.num_layers = len(self.layer_sizes) - 1
        
        self.weights = []
        self.biases = []
       
        for i in range (self.num_layers):
            W = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * 0.01
            b = np.zeros((1, self.layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)
            
        self.activations = [activation_hidden]*(self.num_layers-1) + [linear]
        self.activations_deriv = [activation_hidden_derivative] * (self.num_layers-1) + [linear_derivative]
    
    # === forward pass === #
    def forward(self, X):
        
        self.a_values = [X]
        
        for i in range(self.num_layers):
            z = np.dot(self.a_values[i], self.weights[i]) + self.biases[i]
            a = self.activations[i](z)
            self.a_values.append(a)
        
        return self.a_values[-1]
    
    # === backward pass === #
    def backward(self, X, y_true, loss_derivative):
        
        m = y_true.shape[0]
        self.dW = [None] * self.num_layers
        self.db = [None] * self.num_layers
        
        dA = loss_derivative(y_true, self.a_values[-1])
        
        for i in reversed(range(self.num_layers)):
            dZ = dA * self.activations[i + "_derivative"](self.a_values[i+1])
            self.dW[i] = np.dot(self.a_values[i].T, dZ) / m
            self.db[i] = np.sum(dZ, axis=0, keepdims=True) / m
            dA = np.dot(dZ, self.weights[i].T)
    
    # === update weights === #
    def update_weights(self, lr=0.01):
        for i in range (self.num_layers):
            self.weights[i] -= lr * self.dW[i]
            self.biases[i] -= lr * self.db[i]
    
    # === training loop ===
    def train (self, X, y, epochs=100, lr=0.01, mode='batch', batch_size=32):
        n_samples = X.shape[0]
        losses = []
        
        for epoch in range(epochs):
            if mode == 'batch':
                y_pred = self.forward(X)
                loss = mse(y, y_pred)
                self.backward(y)
                self.update_weights(lr)
                losses.append(loss)
                
            elif mode == 'stochastic':
                loss_epoch = 0
                
                for i in range(n_samples):
                    xi = X[i:i+1]
                    yi = y[i:i+1]
                    y_pred = self.forward(xi)
                    loss = mse(yi, y_pred)
                    self.backward(yi)
                    self.update_weights(lr)
                    loss_epoch += loss
                    
                losses.append(loss_epoch/n_samples)
            
            elif mode == 'minibatch':
                perm = np.random.permutation(n_samples)
                X_shuffled, y_shuffled = X[perm], y[perm]
                loss_epoch = 0
                
                for start in range(0, n_samples, batch_size):
                    end = start + batch_size
                    X_batch = X_shuffled[start:end]
                    y_batch = y_shuffled[start:end]
                    y_pred = self.forward(X_batch)
                    loss = mse(y_batch, y_pred)
                    self.backward(y_batch)
                    self.update_weights(lr)
                    loss_epoch += loss
                    
                losses.append(loss_epoch / (n_samples // batch_size + 1))
            
            else:
                raise ValueError("Mode must be 'batch', 'stochastic', or 'minibatch'")
            
        return losses
    
    # === predict === 
    def predict(self, X):
        return self.forward(X)