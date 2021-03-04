from copy import deepcopy

from IPython.core.display import clear_output
from matplotlib import pyplot as plt

from PictureRecorder import PictureRecorder
from utils import movingaverage


class Plotter(object):
    def __init__(self):

        self.fig = plt.figure(figsize=(25, 7))

        self.recorder = PictureRecorder(self.fig)
        self.gs = self.fig.add_gridspec(10, 10, wspace=0.5)
        self.init_axes()

    def init_axes(self):
        self.ax_train = self.fig.add_subplot(self.gs[:5, :5])
        self.ax_test = self.fig.add_subplot(self.gs[:5, 5:])
        self.ax_latent_space = self.fig.add_subplot(self.gs[6:, 7:])
        self.ax_inter = self.fig.add_subplot(self.gs[6:, 5:7])
        self.ax_loss = self.fig.add_subplot(self.gs[6:, 0:5])

    def clear_axes(self):
        self.ax_train.clear()
        self.ax_test.clear()
        self.ax_latent_space.clear()
        self.ax_inter.clear()
        self.ax_loss.clear()

    def legend(self):
        self.ax_train.legend(['original curve', 'prediction'])
        self.ax_test.legend(['original curve', 'prediction'])
        self.ax_loss.legend(['loss'], loc="upper right")
        self.ax_inter.legend(['log_sigma'], loc="upper right")

    def plot_process(self, current_epoch, losses, epochs, train_curve: dict, test_curve: dict):
        clear_output(wait=True)

        self.init_axes()

        plt1 = self.plot_curve(self.ax_train, train_curve, title="Train Curve")
        plt2 = self.plot_curve(self.ax_test, test_curve, title="Test Curve")

        plt3 = self.plot_latent_space(self.ax_latent_space, train_curve)
        plt4 = self.plot_confidence_interval(self.ax_inter, test_curve)

        plt5 = self.plot_loss(self.ax_loss, current_epoch, epochs, losses, window_size=100)

        self.recorder.add([*plt1, *plt2, *plt3, *plt4, *plt5])

        self.fig.suptitle(f"Training a Conditional Neural Process on 1D functions")

    def save_vid(self, file):
        self.legend()
        self.recorder.save_movie(file)

    def plot_curve(self, curve_ax, curve: dict, title: str = "Curve"):
        data = curve.get('data')
        mu = curve.get('mu')
        log_sigma = curve.get('log_sigma')
        c_points = curve.get('c_points')

        interval = log_sigma

        plots = []
        plots.append(curve_ax.fill_between(data.T[0],
                                           (mu + interval).T[0],
                                           (mu - interval).T[0],
                                           alpha=0.3,
                                           color="#DDDDDD"))

        plots.append(*curve_ax.plot(data.T[0], data.T[1], label="original curve", c="blue"))
        plots.append(*curve_ax.plot(data.T[0], mu, label="prediction", c="green"))

        plots.append(curve_ax.scatter(c_points.T[0],
                                      c_points.T[1],
                                      c="blue",
                                      label="context points"))

        curve_ax.set_ylim([-3, 3])
        curve_ax.set_title(title)
        return plots

    def plot_latent_space(self, ax, curve):
        # Test Latent State

        z = curve.get('z')
        plot = ax.imshow(z, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("Batch Size")
        ax.set_xlabel("Latent Space")
        return [plot]

    def plot_confidence_interval(self, ax, curve):
        data = curve.get('data')
        log_sigma = curve.get('log_sigma')
        c_points = curve.get('c_points')

        plots = [*ax.plot(data.T[0], log_sigma, color="black")]

        for line in c_points.T[0]:
            plots.append(
                ax.axvline(line, color="r"))

        ax.set_ylim([0, 3])
        ax.set_title("Log Sigma")
        return plots

    def plot_loss(self, ax, epoch, epochs, losses, window_size=100):
        plots = [*ax.plot(epochs, losses, alpha=0.5 if epoch > window_size else 1, label="loss", color="red")]
        # Loss function
        if epoch > window_size:
            plots.append(*ax.plot(epochs, movingaverage(losses, window_size), label="moving average", color="blue"))
        ax.set_title("Train Loss")
        return plots
