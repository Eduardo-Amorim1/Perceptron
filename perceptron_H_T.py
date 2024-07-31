import numpy as np
import matplotlib.pyplot as plt

T = np.array([
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 0]
]).flatten()

T_D = np.array([
    [1, 1, 1],
    [1, 1, 0],
    [0, 1, 0]
]).flatten()

H = np.array([
    [1, 0, 1],
    [1, 1, 1],
    [1, 0, 1]
]).flatten()

H_D = np.array([
    [1, 0, 1],
    [1, 1, 1],
    [1, 0, 1]
]).flatten()

labels = {
    'T': 1,
    'H': 0
}

# Dados de entrada
X = np.array([T, H])
y = np.array([labels['T'], labels['H']])

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        # Inicializa o perceptron com o tamanho das entradas e a taxa de aprendizado
        self.weights = np.random.rand(input_size)  # Inicializa os pesos aleatoriamente
        self.bias = np.random.rand()  # Inicializa o bias aleatoriamente
        self.learning_rate = learning_rate  # Define a taxa de aprendizado

    def sigmoid(self, x):
        # Função de ativação sigmoide
        return 1 / (1 + np.exp(-x))  # Calcula a saída da função sigmoide

    def sigmoid_derivative(self, x):
        # Derivada da função sigmoide
        return x * (1 - x)  # Calcula a derivada da sigmoide, usada para ajuste dos pesos

    def predict(self, inputs):
        # Faz uma previsão com base nas entradas
        return self.sigmoid(np.dot(inputs, self.weights) + self.bias)  # Calcula a saída do perceptron

    def train(self, training_inputs, labels, epochs):
        # Treina o perceptron
        errors = []  # Lista para armazenar o erro médio quadrático de cada época
        for epoch in range(epochs):  # Itera por todas as épocas
            total_error = 0  # Inicializa o erro total da época
            for inputs, label in zip(training_inputs, labels):  # Para cada par de entrada e rótulo
                prediction = self.predict(inputs)  # Faz uma previsão
                error = label - prediction  # Calcula o erro (diferença entre o rótulo e a previsão)
                total_error += error**2  # Adiciona o erro quadrático ao erro total
                adjustment = self.learning_rate * error * self.sigmoid_derivative(prediction)  # Calcula o ajuste dos pesos
                self.weights += np.dot(inputs, adjustment)  # Atualiza os pesos
                self.bias += adjustment  # Atualiza o bias
            errors.append(total_error / len(labels))  # Calcula e armazena o erro médio quadrático da época
        return errors  # Retorna a lista de erros ao final do treinamento


# Inicializar o perceptron
input_size = X.shape[1]
print(f'Número de entradas: {input_size}')
perceptron = Perceptron(input_size)

# Treinar o perceptron
epochs = 1000
errors = perceptron.train(X, y, epochs)

# Plotar a curva de erro
plt.plot(errors)
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Curva de erro durante o treinamento')
plt.show()

# Função para imprimir a resposta do perceptron
def test_perceptron(perceptron, inputs, label):
    prediction = perceptron.predict(inputs)
    print(f'Entrada: {label}, Previsão: {prediction:.4f}, Classe: {"T" if prediction >= 0.5 else "H"}')

# Testar o perceptron
test_perceptron(perceptron, T_D, 'T Distorcido')
test_perceptron(perceptron, H_D, 'H Distorcido')
