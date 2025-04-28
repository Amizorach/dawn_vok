import random
import math
import warnings

import matplotlib.pyplot as plt
import torch

class Plotter:
    """Handles interactive plotting during training."""
    def __init__(
        self,
        plot_loss_axis=True,
        plot_results_axis=True,  # Renamed flag
        single_plot_figsize=(7, 5),
        combined_figsize=(12, 5),
        num_samples_to_plot=1  # New parameter for number of random samples
    ):
        """
        Initializes the plot figure and axes based on flags.
        Turns on interactive mode if any plots are generated.

        Args:
            plot_loss_axis (bool): If True, create an axis for loss plotting.
            plot_results_axis (bool): If True, create space for results plotting.
            single_plot_figsize (tuple): Figure size if only one plot area is created.
            combined_figsize (tuple): Figure size if both plot areas are created.
            num_samples_to_plot (int): Number of random samples to plot per batch.
        """
        self.plot_loss_axis = plot_loss_axis
        self.plot_results_axis = plot_results_axis
        self.num_samples_to_plot = num_samples_to_plot

        self.fig = None
        self.ax_loss = None
        self.ax_results_placeholder = None
        self.results_axes = []
        self._interactive_mode = False

        num_areas = plot_loss_axis + plot_results_axis
        if num_areas == 0:
            warnings.warn("PlotUtils initialized with no plots enabled.")
            return

        figsize = combined_figsize if num_areas == 2 else single_plot_figsize
        plt.ion()
        self._interactive_mode = True
        self.fig, main_axes = plt.subplots(1, num_areas, figsize=figsize, squeeze=False)

        idx = 0
        if plot_loss_axis:
            self.ax_loss = main_axes[0, idx]
            idx += 1
        if plot_results_axis:
            self.ax_results_placeholder = main_axes[0, idx]
            self.ax_results_placeholder.set_visible(False)

    def plot_loss(self, loss_history, max_epochs_to_show=200, use_log_scale=True, title="Training Loss"):
        """Plots the training loss history on the loss axis, if it exists."""
        if self.ax_loss is None:
            return
        self.ax_loss.clear()
        lh = min(len(loss_history), max_epochs_to_show)
        current_epoch = len(loss_history)
        self.ax_loss.set_title(f"{title} (Epoch {current_epoch})")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Avg Loss")
        self.ax_loss.grid(True)
        if lh == 0:
            return
        start = max(1, current_epoch - lh + 1)
        xs = range(start, current_epoch + 1)
        ys = loss_history[-lh:]
        self.ax_loss.plot(xs, ys)
        self.ax_loss.set_yscale('log' if use_log_scale else 'linear')

    def plot_results(self, gts, preds, grid_layout=None, main_title="Results Comparison"):
        """
        Randomly samples up to `self.num_samples_to_plot` entries from the batch
        and plots ground-truth vs predictions in a grid layout.
        """
        if not self.plot_results_axis or self.fig is None:
            return

        # 1) REMOVE old result axes entirely
        for ax in self.results_axes:
            self.fig.delaxes(ax)
        self.results_axes.clear()

        # 2) CLEAR and HIDE the placeholder
        self.ax_results_placeholder.clear()
        self.ax_results_placeholder.set_visible(False)

        # 3) VALIDATE inputs
        if not (isinstance(gts, torch.Tensor) and isinstance(preds, torch.Tensor)):
            print("Warning: plot_results expects PyTorch tensors for gts and preds.")
            self.ax_results_placeholder.set_visible(True)
            self.ax_results_placeholder.text(0.5, 0.5, 'Invalid Input',
                                             ha='center', va='center', fontsize=12)
            return

        batch_size = gts.shape[0]
        if batch_size == 0 or preds.shape[0] == 0:
            print("Warning: gts or preds tensor is empty.")
            self.ax_results_placeholder.set_visible(True)
            self.ax_results_placeholder.text(0.5, 0.5, 'Empty Input',
                                             ha='center', va='center', fontsize=12)
            return

        # 4) SAMPLE indices
        n_plot = min(self.num_samples_to_plot, batch_size)
        indices = random.sample(range(batch_size), k=n_plot)

        # 5) GRID layout
        if grid_layout:
            rows, cols = grid_layout
            if rows * cols < n_plot:
                warnings.warn(f"Grid {grid_layout} too small for {n_plot} samples, adjusting cols.")
                cols = math.ceil(n_plot / rows)
        else:
            rows = 1 if n_plot <= 2 else 2
            cols = math.ceil(n_plot / rows)

        # 6) MAKE a fresh GridSpec for just the results region
        #    Adjust top/bottom/left/right as needed to leave room for loss axis
        gs = self.fig.add_gridspec(
            nrows=rows, ncols=cols,
            top=0.90, bottom=0.10, left=0.55, right=0.98,
            wspace=0.4, hspace=0.4
        )

        # 7) CREATE each subplot in the new GridSpec
        for i, idx in enumerate(indices):
            r, c = divmod(i, cols)
            ax = self.fig.add_subplot(gs[r, c])
            self.results_axes.append(ax)

            gt = gts[idx].detach().cpu().numpy()
            pr = preds[idx].detach().cpu().numpy()

            ax.plot(gt, linestyle='--', marker='.', markersize=4, label='Ground-Truth')
            ax.plot(pr, linestyle='-',  marker='.', markersize=4, label='Prediction')
            ax.set_title(f"Sample {idx}", fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.grid(True, linestyle=':', linewidth=0.5)
            if r == rows - 1:
                ax.set_xlabel("Dim", fontsize=7)
            if c == 0:
                ax.set_ylabel("Val", fontsize=7)
            if i == 0:
                ax.legend(fontsize=6, loc='upper right')

        # 8) FINISH
        self.fig.tight_layout()

    def update(self):
        """Redraw canvas."""
        if not self.fig:
            return
        self.fig.canvas.draw_idle()
        plt.pause(0.01)

    def finalize(self, wait=False):
        """Finalize interactive plot."""
        if not self.fig:
            return
        self.fig.canvas.draw_idle()
        if self._interactive_mode:
            plt.ioff()
            self._interactive_mode = False
        if wait:
            plt.show()
        else:
            plt.close()

    def save_plot(self, filename="plot.png"):
        """Save figure to file."""
        if not self.fig:
            warnings.warn("Save attempted with no figure.")
            return
        try:
            self.fig.savefig(filename, dpi=150)
            print(f"Plot saved to {filename}")
        except Exception as e:
            print(f"Error saving plot: {e}")
