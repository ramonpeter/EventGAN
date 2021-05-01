""" Loss plots """

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("agg")

# pylint: disable=W0702
try:
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
except:
    print("No latex installed")


def plot_loss(loss, log_dir=".", name=""):
    """Plot the traings curve"""
    fig, ax1 = plt.subplots(1, figsize=(10, 4))
    epoch = np.arange(len(loss))
    loss = np.array(loss)

    if name == "C":
        try:
            plt.plot(epoch, loss[:, 0], color="red", markersize=12, label=r"Total")
            plt.plot(
                epoch,
                loss[:, 0] - loss[:, 1],
                color="green",
                markersize=12,
                label=r"BC Loss",
                linestyle="dashed",
            )
            plt.plot(
                epoch,
                loss[:, 1],
                color="royalblue",
                markersize=12,
                label=r"Gradient Penalty",
                linestyle="dashed",
            )
        except:
            plt.plot(epoch, loss[:, 0], color="red", markersize=12, label=r"BC Loss")
    else:
        try:
            plt.plot(epoch, loss[:, 0], color="red", markersize=12, label=r"Total")
            plt.plot(
                epoch,
                loss[:, 0] - loss[:, 1],
                color="green",
                markersize=12,
                label=r"BC Loss",
                linestyle="dashed",
            )
            plt.plot(
                epoch,
                loss[:, 1],
                color="royalblue",
                markersize=12,
                label=r"MMD Loss",
                linestyle="dashed",
            )
        except:
            plt.plot(epoch, loss[:, 0], color="red", markersize=12, label=r"BC Loss")

    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=9,
        fancybox=True,
        shadow=True,
        prop={"size": 10},
    )
    ax1.set_xlabel(r"Epochs")
    ax1.set_ylabel(r"Loss")
    fig.savefig(f"{log_dir}/{name}_Loss.pdf", dpi=120, bbox_inches="tight")
    plt.close("all")
