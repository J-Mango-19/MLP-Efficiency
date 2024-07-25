import subprocess
import os

class File():
    def __init__(self, name, directory, file_type, arguments):
        self.name = name
        self.dir = directory
        self.type = file_type
        self.args = arguments


def run_program(file_name, dir_name, file_type, start_dir, args):
    print(f"Running {dir_name}/{file_name}...")
    executable_path = os.path.join(dir_name, file_name)
    os.chdir(os.path.dirname(os.path.abspath(executable_path)))

    if file_type == 'python':
        result = subprocess.run(['python3', file_name] + args, capture_output=True, text=True)
    elif file_type == 'C':
        result = subprocess.run([f'./{file_name}'] + args, capture_output=True, text=True)
    elif file_type == 'Make':
        pass
    if result.returncode != 0:
        print(f'{script_name} encountered errors:', result.stderr)

    os.chdir(start_dir)
    return result.stdout.splitlines()



def main():
    file_path = os.path.abspath(__file__)
    start_dir = os.path.dirname(file_path)
    main_py = File('main.py', 'py_mnist', 'python', '-nodisplay')
    C_optimized = File('mnist_nn', 'optimized', 'C', '-nodisplay')
    C_base = File('mnist_nn', 'base', 'C', '-nodisplay')

    file_names = ['main.py', 'mnist_nn', 'mnist_nn']
    directories = ['py_mnist', 'optimized', 'base']
    file_types = ['python', 'C', 'C']
    arguments = ['-nodisplay']

    for file_name, directory, file_type in zip(file_names, directories, file_types):
        out = run_program(file_name, directory, file_type, start_dir, arguments)
        for line in out[-3:]:
            print(line)
        print('\n')

if __name__ == '__main__':
    main()


