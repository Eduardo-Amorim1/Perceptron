import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iterations=1000, initial_weights=None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = initial_weights
        self.bias = 0

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Inicializando os pesos se não forem fornecidos
        if self.weights is None:
            self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)

                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = [self.activation_function(i) for i in linear_output]
        return np.array(y_predicted)

# Definindo as entradas do XOR
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Definindo a saída correspondente do XOR
y = np.array([0, 1, 1, 0])

# Aplicando a transformação polinomial
X_transformed = np.column_stack((X[:, 0], X[:, 1], X[:, 0] * X[:, 1]))

# Inicializando com pesos personalizados
initial_weights = np.array([1, 0, 0.5])  # Pesos personalizados
perceptron = Perceptron(learning_rate=0.1, n_iterations=1000, initial_weights=initial_weights)
perceptron.fit(X_transformed, y)

# Fazendo previsões
predictions = perceptron.predict(X_transformed)
print("Previsões:", predictions)

# Testando com novos dados
new_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
new_data_transformed = np.column_stack((new_data[:, 0], new_data[:, 1], new_data[:, 0] * new_data[:, 1]))
new_predictions = perceptron.predict(new_data_transformed)
print("Previsões para novos dados:", new_predictions)
