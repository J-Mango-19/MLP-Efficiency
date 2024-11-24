import subprocess
import os
import sys
import glob
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    visualize = True # create bar plots visualizing training times
    dataset = "FMNIST" # can be substituted with MNIST

    if subprocess.run(['bash', f'./bin/{dataset}_download.sh']).returncode != 0:
        sys.exit(1)

    if len(sys.argv) == 1:
        arguments = ['-nodisplay', '-iterations', '10000'] 
    else:
        arguments = sys.argv[1:]

    root_dir = Path.cwd() 
    sub_dirs = [d.name for d in root_dir.iterdir() if d.is_dir()]

    times_dict = {}
    for sub_dir in sub_dirs:
        os.chdir(sub_dir)
        # skip directories with nothing to run
        if not (run_file := glob.glob("run_*.sh")):
            os.chdir("..")
            continue

        command = ['bash', run_file[0]]
        command += arguments
        print(f"Running {command} in directory: {sub_dir}")
        output = subprocess.run(command)

        if output.returncode != 0:
            print(f"Failed to run {command} in directory: {sub_dir}")
            sys.exit(1)

        with open("stats.txt", "r") as file:
            times_dict[run_file[0]] = file.read().splitlines()

        if Path("stats.txt").exists():
            os.remove("stats.txt")

        os.chdir("..")

    # Everything after this is plotting times. If not plotting, then done.
    if visualize is False: 
        sys.exit(0)

    # Plotting
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8))
    plt.subplots_adjust(wspace=0.25)

    x = range(len(times_dict))

    names = [k for k in times_dict.keys()]

    alloc_times = [float(l[0]) for l in times_dict.values()]
    train_times = [float(l[1]) for l in times_dict.values()]
    inference_times = [float(l[2]) for l in times_dict.values()]


    alloc_bar = ax1.bar(x, alloc_times, label = "Allocation Time")
    train_bar = ax1.bar(x, train_times, bottom=alloc_times, label="Training Time") 

    ax1.set_title("Total Training time (10,000 training steps) by implementation")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, ha="right")
    ax1.set_xlabel("MLP Implementation")
    ax1.set_ylabel("Time (seconds)")
    ax1.legend()

    # For the first subplot (ax1)
    for i, (bar1, bar2) in enumerate(zip(alloc_bar, train_bar)):
        height1, height2 = bar1.get_height(), bar2.get_height()
        total_height = height1 + height2
        ax1.text(bar1.get_x() + bar1.get_width() / 2, height1 / 2, f"Allocation time: {height1:.3f}",
                 ha="center", va="center")
        ax1.text(bar1.get_x() + bar1.get_width() / 2, height1 + height2/2, f"Training time: {height2:.3f}",
                 ha="center", va="center")

    inference_bar = ax2.bar(x, inference_times)
    ax2.set_title("Full batch inference time by implementation")
    ax2.set_ylabel("Inference time (seconds)")
    ax2.set_xlabel("MLP Implementation")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, ha = "right")

    # Annotate bars with their values
    for bar in inference_bar:
        yval = bar.get_height()
        ax2.text(
        bar.get_x() + bar.get_width() / 2,
        yval, 
        f'{yval:.3f}',
        ha="center",
        va="bottom"
        )

    plt.tight_layout()

    print("Saving results in 'training_time_chart.png'")
    figure.savefig("training_time_chart.png", format='png', bbox_inches='tight')

if __name__ == "__main__":
    main()
