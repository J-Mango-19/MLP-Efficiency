import numpy as np

def backward(x, y, w1, w2, w3, z1, a1, z2, a2, z3, a3):
    m = y.size
    one_hot_y = one_hot(y)
    dz3 = a3 - one_hot_y
    dw3 = (1/m) * dz3.dot(a2.t)
    db3 = (1/m) * np.sum(dz3)
    da2 = w3.t.dot(dz3)
    dz2 = da2 * deriv_relu(z2)
    dW2 = (1/m) * dZ2.dot(A1.T)
    db2 = (1/m) * np.sum(dZ2)
    dA1 = W2.T.dot(dZ2)
    dZ1 = dA1 * deriv_relu(Z1)
    dW1 = (1/m) * dZ1.dot(X.T)
    db1 = (1/m) * np.sum(dZ1)
    return dW1, db1, dW2, db2, dW3, db3

def relu(z):
    return np.maximum(z, 0)

def softmax(z):
    return np.exp(z) / sum(np.exp(z))

def forward(x, w1, b1, w2, b2, w3, b3):
    z1 = w1.dot(x) + b1
    a1 = relu(z1)
    z2 = w2.dot(a1) + b2
    a2 = relu(z2)
    z3 = w3.dot(a2) + b3
    a3 = softmax(z3)
    return z1, a1, z2, a2, z3, a3

def deriv_relu(a):
    return a > 0


