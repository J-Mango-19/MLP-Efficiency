import numpy as np
import time
import sys
from neural_network import train, forward, get_predictions
from utils import get_input, display_output

def main():
    # load data, hyperparameters, and other preferences
    lr, batch_size, steps, display_start, display_end, status_interval, file_path = get_input(sys.argv)
    start_alloc = time.time() # allocation and training time are separated to isolate training performance for comparison 
    print("Reading MNIST data...")
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1).T 

    test_data = data[:, :1000]
    train_data = data[:, 1000:]

    Y_test = test_data[0]
    Y_train = train_data[0]

    X_test = test_data[1:] / 255
    X_train = train_data[1:] / 255
    end_alloc = time.time()
    
    # training
    train_start_time = time.time()
    W1, b1, W2, b2, W3, b3 = train(X_train, Y_train, X_test, Y_test, lr, steps, batch_size, status_interval)
    train_end_time = time.time()
    print(f"Allocation time: {end_alloc - start_alloc:.4f} seconds")
    print(f"Training time: {train_end_time - train_start_time:.4f} seconds")

    # Full batch inference time isolated for comparison 
    start_forward = time.time()
    _, _, _, _, _, A3 = forward(X_train, W1, b1, W2, b2, W3, b3)
    predictions = get_predictions(A3)
    end_forward = time.time()
    print(f'One inference of entire training set: {end_forward - start_forward:.4f} seconds') # Inference time isolated for comparison

    display_output(X_test, Y_test, display_start, display_end, W1, b1, W2, b2, W3, b3)

if __name__ == '__main__':
    main()
