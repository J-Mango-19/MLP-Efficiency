import sys 
import numpy as np
import matplotlib.pyplot as plt

def usage(code):
    print("Usage:")
    print("python3 mnist_nn -<flag1> -<flag2>")
    print("flags:")
    print("     -h for help")
    print("     -lr sets learning rate")
    print("     -batch_size sets batch size")
    print("     -iterations sets number of training iterations")
    print("     -display <num1> <num2> displays digits and their predictions from the test dataset")
    print("ensure 0 < num1 < num2 < 1000 as there are 1000 test dataset digits")
    print("     -nodisplay turns the automatic display examples off")
    print("     -status_interval sets the interval at which the test and training accuracy will be displayed")
    print("     -mode sets the type of output (labeled times or unlabeled times)")
    print("example usage: python3 mnist_nn -lr 0.05 -status_interval 200 -display 110 120")
    sys.exit(code)

def get_input(arguments):
    lr = 0.1
    batch_size = 100
    steps = 10000
    display_start = 100
    display_end = 105
    status_interval = 100
    arguments = arguments[1:]
    file_path = '../data/MNIST_data.csv'
    mode = 0
    while arguments:
        arg = arguments.pop(0)
        if arg == "-h":
            usage(0)
        elif arg == "-lr":
            lr = int(arguments.pop(0))
        elif arg == "-iterations":
            steps = int(arguments.pop(0))
        elif arg == "-batch_size":
            batch_size = int(arguments.pop(0))
        elif arg == "-status_interval":
            status_interval = int(arguments.pop(0))
        elif arg == "-display":
            display_start = int(arguments.pop(0))
            display_end = int(arguments.pop(0))
        elif arg == "-nodisplay":
            display_start = 0
            display_end = 0
        elif arg == "-data_collection_mode":
            mode = 1
        else:
            usage(1)

    if not (display_start <= display_end and display_start >= 0 and display_end < 1000):
        usage(1)

    return lr, batch_size, steps, display_start, display_end, status_interval, file_path, mode

def display_output(X_test, Y_test, start_idx, end_idx, W1, b1, W2, b2, W3, b3):
    from neural_network import forward, get_predictions
    for i in range(start_idx, end_idx):
        image = X_test[:, i]
        label = int(Y_test[i])
        _, _, _, _, _, A3 = forward(image, W1, b1, W2, b2, W3, b3)
        prediction = get_predictions(A3)[0]
        plt.title(f"Prediction: {prediction} Label: {label}")
        image = image.reshape(28, 28) * 255 # undoing normalization
        plt.gray()
        plt.imshow(image, interpolation='nearest')
        plt.savefig(f'example{i}.png', format='png', bbox_inches='tight', pad_inches=0)

def he_init(shape, fan_in):
    n = np.prod(shape)
    u1 = np.random.random(n)
    u2 = np.random.random(n)
    z1 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    std_dev = np.sqrt(2.0 / fan_in)
    return (z1 * std_dev).reshape(shape)

def init_params(hidden_nodes_1=30, hidden_nodes_2=20):
    num_in = 784 # flattened 28x28 image inputs
    num_out = 10 # 10 output classes, 0-9
    w1 = he_init((hidden_nodes_1, num_in), num_in)
    b1 = he_init((hidden_nodes_1, 1), num_in)
    w2 = he_init((hidden_nodes_2, hidden_nodes_1), hidden_nodes_1)
    b2 = he_init((hidden_nodes_2, 1), hidden_nodes_1)
    w3 = he_init((num_out, hidden_nodes_2), hidden_nodes_2)
    b3 = he_init((num_out, 1), hidden_nodes_2)
    return w1, b1, w2, b2, w3, b3

def get_accuracy(predictions, y):
    return np.sum(predictions == y) / y.size

def one_hot(y):
    one_hot_y = np.zeros((10, y.size))
    for i in range(one_hot_y.shape[0]):
        for j in range(one_hot_y.shape[1]):
          if i == y[j]:
            one_hot_y[i][j] = 1
    return one_hot_y

def display_times(alloc_time, train_time, inference_time, mode):
    if mode == 0:
        print(f"Allocation time: {alloc_time:.4f} seconds")
        print(f"Training time: {train_time:.4f} seconds")
        print(f'One inference of entire training set (784 pixels x 41000 examples): {inference_time:.4f} seconds') # Inference time isolated for comparison
    elif mode == 1:
        print(f'{alloc_time:.4f}')
        print(f'{train_time:.4f}')
        print(f'{inference_time:.4f}')
