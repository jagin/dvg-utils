import numpy as np


def plot_metrics(args):
    from matplotlib import pyplot as plt

    # Plot metrics
    fig, ax = plt.subplots(dpi=128, figsize=(10, 6))
    color_idx = np.linspace(0, 1, len(args["input"]))
    title = None
    ylabel = None
    for i, metrics_file in zip(color_idx, args["input"]):
        metrics = np.loadtxt(metrics_file)
        iterations = np.arange(1, len(metrics) + 1)
        if args["chart"] == "iter":
            metrics = metrics[:, 0] * 1000
            title = "Iteration execution time"
            ylabel = "[ms]"
        elif args["chart"] == "ips":
            metrics = metrics[:, 1]
            title = "Averaged iterations per second"
            ylabel = "[it/s]"
        elif args["chart"] == "spi":
            metrics = metrics[:, 2] * 1000
            title = "Averaged milliseconds per iteration"
            ylabel = "[ms/it]"
        mean = [np.mean(metrics)] * len(metrics)
        ax.plot(iterations, metrics, label=f"{metrics_file} data", color=plt.cm.cool(i))
        ax.plot(iterations, mean, label=f"{metrics_file} mean", linestyle="--", color=plt.cm.cool(i))
        ax.set_xlim(1, len(metrics) + 1)

    # Format Plot
    plt.grid(True)
    plt.title(title, fontsize=24)
    plt.xlabel("[it]", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.legend(loc='upper right')
    plt.show()
