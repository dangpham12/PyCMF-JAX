import matplotlib.pyplot as plt
import json
import numpy as np
def get_data(shape, type):
    path = f"../data/{shape}_{type}_devicesync.json" # Add/remove devicesync path
    with open(path, "r") as f:
        data = json.load(f)

    backends = data.keys()
    times = [data[backend]["total"] for backend in backends]
    mean_times = [data[backend]["mean"] for backend in backends]
    return backends, times, mean_times

def get_data_gpu(shape, type):
    path = f"../data/{shape}_{type}.json"
    with open(path, "r") as f:
        data = json.load(f)

    desired_backends = ["gt:gpu", "dace:gpu", "cuda"]
    backends = [backend for backend in data.keys() if backend in desired_backends] # Uncomment this line for GPU backends
    times = [data[backend]["total"] for backend in backends]
    mean_times = [data[backend]["mean"] for backend in backends]
    return backends, times, mean_times

if __name__ == "__main__":
    canvas = [50, 400]
    data_type = ["float64", "float32"]
    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]

    backends, times, mean_times = get_data(canvas[0], data_type[1])
    backends2, times2, mean_times2 = get_data(canvas[1], data_type[1])

    bar_width = 0.4
    x = np.arange(len(backends))

    # plt.bar(x, mean_times, width=bar_width, color=colors) # Uncomment this line to plot times/backend

    plt.bar(x - bar_width / 2, times, width=bar_width, label='50x50')
    plt.bar(x + bar_width / 2, times2, width=bar_width, label="400x400") # Uncomment these lines to compare


    plt.xlabel('Backend')
    plt.ylabel('Time (seconds)')
    plt.title(f'Iteration step time {data_type[1]} 50x50 vs 400x400')
    plt.xticks(x, backends, fontsize=6, rotation=45)
    plt.legend()

    plt.show()