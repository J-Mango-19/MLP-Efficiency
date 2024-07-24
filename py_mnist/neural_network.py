import numpy as np
from utils import init_params, get_accuracy, one_hot

def get_predictions(a3):
    return np.argmax(a3, 0)

def relu(z):
    return np.maximum(z, 0)

def softmax(z):
    return np.exp(z) / sum(np.exp(z))

def deriv_relu(a):
    return a > 0

def forward(x, w1, b1, w2, b2, w3, b3):
    z1 = w1.dot(x) + b1
    a1 = relu(z1)
    z2 = w2.dot(a1) + b2
    a2 = relu(z2)
    z3 = w3.dot(a2) + b3
    a3 = softmax(z3)
    return z1, a1, z2, a2, z3, a3

def backward(X, Y, W1, W2, W3, Z1, A1, Z2, A2, Z3, A3):
    m = Y.size
    one_hot_y = one_hot(Y)
    dZ3 = A3 - one_hot_y
    dW3 = (1/m) * dZ3.dot(A2.T)
    db3 = (1/m) * np.sum(dZ3)
    dA2 = W3.T.dot(dZ3)
    dZ2 = dA2 * deriv_relu(Z2)
    dW2 = (1/m) * dZ2.dot(A1.T)
    db2 = (1/m) * np.sum(dZ2)
    dA1 = W2.T.dot(dZ2)
    dZ1 = dA1 * deriv_relu(Z1)
    dW1 = (1/m) * dZ1.dot(X.T)
    db1 = (1/m) * np.sum(dZ1)
    return dW1, db1, dW2, db2, dW3, db3

def update_weights(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, lr):
    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2
    W3 = W3 - lr * dW3
    b3 = b3 - lr * db3
    return W1, b1, W2, b2, W3, b3

def train(X_train, Y_train, X_test, Y_test, lr, steps, batch_size, status_interval):
  W1, b1, W2, b2, W3, b3 = init_params()
  pos_1 = -1 * batch_size

  for step in range(steps):
    pos_1 += batch_size
    pos_2 = pos_1 + batch_size
    if pos_2 > X_train.shape[1]:
      pos_1 = 0
      pos_2 = pos_1 + batch_size
    X_in = X_train[:, pos_1:pos_2]
    Y_in = Y_train[pos_1:pos_2]

    Z1, A1, Z2, A2, Z3, A3 = forward(X_in, W1, b1, W2, b2, W3, b3)
    dW1, db1, dW2, db2, dW3, db3 = backward(X_in, Y_in, W1, W2, W3, Z1, A1, Z2, A2, Z3, A3)
    W1, b1, W2, b2, W3, b3 = update_weights(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, lr)

    if step % status_interval == 0:
        _, _, _, _, _, A3 = forward(X_train, W1, b1, W2, b2, W3, b3)
        predictions = get_predictions(A3)
        train_acc = get_accuracy(predictions, Y_train)
        _, _, _, _, _, A3 = forward(X_test, W1, b1, W2, b2, W3, b3)
        predictions = get_predictions(A3)
        test_acc = get_accuracy(predictions, Y_test)
        print(f'Steps: {step:4d} | Train Accuracy: {train_acc:5.3f} | Test Accuracy : {test_acc:5.3f}')

  return W1, b1, W2, b2, W3, b3

