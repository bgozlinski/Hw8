import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

num_classes = 10 # całkowita liczba klas, w naszym przypadku są to liczby od 0 do 9
num_features = 784 # liczba atrybutów wektora wejściowego 28 * 28 = 784

learning_rate = 0.001 # szybkość uczenia się sieci neuronowej
training_steps = 3000 # maksymalna liczba epok
batch_size = 256 # przeliczymy wagi sieci nie na całej próbce, ale na jej losowym podzbiorze elementów bat
display_step = 100 # co 100 iteracji pokażemy aktualną wartość funkcji straty i dokładności

n_hidden_1 = 128 # liczba neuronów warstwy 1
n_hidden_2 = 256 # liczba neuronów warstwy 2

from tensorflow.keras.datasets import mnist

# Ładowanie zestawu danych
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Konwersja pikseli całkowitych na typ float32
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)

# Konwertujemy macierze 28x28 pikseli na wektor składający się z 784 elementów
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])

# Normalizacja wartości pikseli
x_train, x_test = x_train / 255., x_test / 255.

# Zmiksujmy dane treningowe
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)


# Stwórzmy sieć neuronową
class DenseLayer(tf.Module):
    def __init__(self, in_features, out_features, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(tf.random.normal([in_features, out_features]), name='w')
        self.b = tf.Variable(tf.zeros([out_features]), name='b')

    def __call__(self, x):
        x = tf.matmul(x, self.w) + self.b
        return tf.nn.sigmoid(x)


class NN(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.layer1 = DenseLayer(num_features, n_hidden_1)
        self.layer2 = DenseLayer(n_hidden_1, n_hidden_2)
        self.out_layer = DenseLayer(n_hidden_2, num_classes)

    def __call__(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.out_layer(x)
        return tf.nn.softmax(x)


# W tym przypadku wygodnie jest przyjąć entropię krzyżową jako funkcję błędu
def cross_entropy(y_pred, y_true):
    # Encode label to a one hot vector.
    y_true = tf.one_hot(y_true, depth=num_classes)

    # Clip prediction values to avoid log(0) error.
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)

    # Obliczanie entropii krzyżowej
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))


# Jako miernik jakości stosujemy dokładność
def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Stwórzmy przykładową sieć neuronową
neural_net = NN(name="mnist")


# Funkcja treningu sieci neuronowej
def train(nn, input_x, output_y):
  # Do dopasowania wag sieci wykorzystamy stochastyczne zejście gradientowe
  optimizer = tf.optimizers.SGD(learning_rate)

  # Aktywacja automatycznego różnicowania
  with tf.GradientTape() as g:
    pred = neural_net(input_x)
    loss = cross_entropy(pred, output_y)

    # Utwórz zoptymalizowaną listę parametrów
    trainable_variables = nn.trainable_variables

    # Oblicz na ich podstawie wartość gradientu
    gradients = g.gradient(loss, trainable_variables)

    # Zmodyfikuj parametry
    optimizer.apply_gradients(zip(gradients, trainable_variables))


# Szkolenie sieciowe
loss_history = []  # każdy krok display_step zapisuje bieżący błąd sieci neuronowej na tej liście
accuracy_history = [] # każdy krok display_step zapisuje aktualną dokładność sieci neuronowej na tej liście

# W tej pętli będziemy trenować sieć neuronową
# Z treningowego zbioru danych train_data, wyodrębnij losowy podzbiór, na którym
# będzie trenowana. Użyj metody take dostępnej dla szkoleniowego zbioru danych.
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # Aktualizacja wag sieci neuronowej
    train(neural_net, batch_x, batch_y)

    if step % display_step == 0:
        pred = neural_net(batch_x)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        loss_history.append(loss)
        accuracy_history.append(acc)
        print(f"Step: {step}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")


# Wykreśl zmianę dokładności i strat jako funkcję skoku
# Jeśli zostanie to zrobione poprawnie, dokładność powinna wzrosnąć, a straty powinny się zmniejszyć.

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(display_step, training_steps + 1, display_step), loss_history)
plt.title('Model Loss')
plt.xlabel('Step')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(range(display_step, training_steps + 1, display_step), accuracy_history)
plt.title('Model Accuracy')
plt.xlabel('Step')
plt.ylabel('Accuracy')

plt.show()


# Oblicz dokładność wytrenowanej sieci neuronowej

pred = neural_net(x_test)
print(f"Test Accuracy: {accuracy(pred, y_test):.4f}")


# Przetestuj wytrenowaną sieć neuronową na 10 obrazach. Z próbki testowej wybierz 5
# losowych obrazów i wprowadź je do sieci neuronowej.
# Wyprowadź obraz i zapisz obok niego odpowiedź sieci neuronowej.
# Czy sieć neuronowa się myli, a jeśli tak, to jak często?

import random

samples = random.sample(range(x_test.shape[0]), 10)
plt.figure(figsize=(10, 5))
for i, idx in enumerate(samples):
    img = x_test[idx].reshape((28, 28))
    pred_label = np.argmax(pred[idx])
    true_label = y_test[idx]
    plt.subplot(2, 5, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Pred: {pred_label}, True: {true_label}")
    plt.axis('off')
plt.show()