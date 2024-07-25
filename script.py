import subprocess
import os
import sys
import matplotlib.pyplot as plt

class File():
    def __init__(self, name, directory, file_type, arguments, display_name):
        self.file_name = name
        self.dir = directory
        self.type = file_type
        self.args = arguments
        self.display_name = display_name

def run_program(_file, start_dir, suppress_output):
    file_path = os.path.join(_file.dir, _file.file_name)
    os.chdir(os.path.dirname(os.path.abspath(file_path)))

    if _file.type == 'python':
        print(f"Running {_file.dir}/{_file.file_name}...")
        result = subprocess.run(['python3', _file.file_name] + _file.args, capture_output=suppress_output, text=True)
    elif _file.type == 'C':
            print(f"Running {_file.dir}/{_file.file_name}...")
            try:
                result = subprocess.run([f'./{_file.file_name}'] + _file.args, capture_output=suppress_output, text=True)
            except FileNotFoundError:
                print(f"Executable not found: Compiling {_file.dir}/{_file.file_name}")
                result = subprocess.run(['make'], capture_output=True, text=True)
                print(f"Running {_file.dir}/{_file.file_name}...")
                result = subprocess.run([f'./{_file.file_name}'] + _file.args, capture_output=suppress_output, text=True)

    if result.returncode != 0:
        print(f'{_file.file_name} encountered errors:', result.stderr)

    os.chdir(start_dir)
    try: 
        return result.stdout.splitlines()
    except AttributeError:
        pass

def visualize_data(files):
    # Prepare data
    names = [file.display_name for file in files]
    alloc_times = [float(file.alloc_time) for file in files]
    train_times = [float(file.train_time) for file in files]
    inference_times = [float(file.inference_time) for file in files]
    x = range(len(names))

    # Set up the figures
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    # plot the alloc & train times
    ax1.bar(x, alloc_times, label='Allocation time')
    ax1.bar(x, train_times, label='Training time', bottom=alloc_times)
    ax1.set_title('Total time by implementation')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_xlabel('Implementation')
    ax1.set_ylabel('Time (seconds)')
    ax1.legend()
    plt.show()

    bars = ax2.bar(x, inference_times)
    ax2.set_title("Full batch inference time by implementation")
    ax2.set_ylabel('Inference time')
    ax2.set_xlabel('Implementation')
    ax2.set_xticks(x, names)

    # Annotate bars with the actual values
    for bar in bars:
        yval = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            f'{yval:.3f}',  # Format the value to three decimal places
            ha='center',
            va='bottom'  # Align text to the bottom of the bar
        )
    plt.show()

    # Save figures
    fig1.savefig('total_times_plot.png', format='png')
    fig2.savefig('inference_times_plot.png', format='png')

def download_dataset():
    # download dataset if it is not already in ./data
    file_path = os.path.join(os.getcwd(), "data", "MNIST_data2.csv")
    if os.path.exists(file_path):
        print("MNIST dataset already downloaded")
    else:
        print("MNIST dataset not found")
        subprocess.run(['bash', 'download.sh'])

def main():
    file_path = os.path.abspath(__file__)
    start_dir = os.path.dirname(file_path)
    suppress_output = True if '-visualize' in sys.argv else False

    download_dataset()

    arguments = ['-nodisplay', '-iterations', '10000']
    if suppress_output:
        arguments.append('-data_collection_mode')

    files = [File('main.py', 'numpy_nn', 'python', arguments, 'Numpy neural net'), 
            File('mnist_nn', 'C_base_nn', 'C', arguments, 'C base neural net'), 
            File('mnist_nn', 'C_optimized_nn', 'C', arguments, 'C optimized neural net')]

    for _file in files:
        out = run_program(_file, start_dir, suppress_output)
        if suppress_output == True:
            _file.alloc_time = out[-3]
            _file.train_time = out[-2]
            _file.inference_time = out[-1]

    if suppress_output == True:
        visualize_data(files)

if __name__ == '__main__':
    main()


