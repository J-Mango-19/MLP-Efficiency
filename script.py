import subprocess
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg')

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
                make_result = subprocess.run(['make'], capture_output=True, text=True)
                if make_result.returncode == 0:
                    print(f"Running {_file.dir}/{_file.file_name}...")
                    result = subprocess.run([f'./{_file.file_name}'] + _file.args, capture_output=suppress_output, text=True)
                else:
                    print(f'{_file.file_name} encountered errors:', make_result.stderr)
                    os.chdir(start_dir)
                    return

    os.chdir(start_dir)

    if result.returncode != 0:
        print(f'{_file.file_name} encountered errors:', result.stderr)
        return

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    plt.subplots_adjust(wspace=0.25)

    # plot the alloc & train times
    bars1 = ax1.bar(x, alloc_times, label='Allocation time')
    bars2 = ax1.bar(x, train_times, label='Training time', bottom=alloc_times)
    ax1.set_title('Total time by implementation')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, ha='right')
    ax1.set_xlabel('Neural Net Implementation')
    ax1.set_ylabel('Time (seconds)')
    ax1.legend()

    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax1.text(bar1.get_x() + bar1.get_width() / 2, height1, f'Allocation time: {height1:.3f}',
                 ha='center', va='bottom')
        ax1.text(bar2.get_x() + bar2.get_width() / 2, height1 + height2, f'Training time: {height2:.3f}',
                 ha='center', va='bottom')

    bars = ax2.bar(x, inference_times)
    ax2.set_title("Full batch inference time by implementation")
    ax2.set_ylabel('Inference time (seconds)')
    ax2.set_xlabel('Neural Net Implementation')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, ha='right')

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

    plt.tight_layout()
    #plt.show()

    # Save figures
    fig.savefig('combined_plot.png', format='png', bbox_inches='tight')

def download_dataset():
    # download dataset if it is not already in ./data
    file_path = os.path.join(os.getcwd(), "data", "MNIST_data.csv")
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

    files = [File('main.py', 'numpy_nn', 'python', arguments, 'Python (Numpy)'), 
            File('mnist_nn', 'C_base_nn', 'C', arguments, 'C (base)'), 
            File('mnist_nn', 'C_optimized_nn', 'C', arguments, 'C (optimized)'),
            File('mnist_nn', 'CBLAS_nn', 'C', arguments, 'C (CBLAS)')]

    for _file in files:
        out = run_program(_file, start_dir, suppress_output)
        if suppress_output == True:
            try: 
                _file.alloc_time = out[-3]
                _file.train_time = out[-2]
                _file.inference_time = out[-1]
            except TypeError: # meaning that this file didn't compile
                files.remove(_file) # don't display results from the file that didn't compile


    if suppress_output == True:
        visualize_data(files)

if __name__ == '__main__':
    main()

