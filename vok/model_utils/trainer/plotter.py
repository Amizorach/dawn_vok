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
        num_samples_to_plot=4  # New parameter for number of random samples
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

        # Clear previous result axes
        for ax in self.results_axes:
            ax.remove()
        self.results_axes.clear()

        # Validate tensors
        if not isinstance(gts, torch.Tensor) or not isinstance(preds, torch.Tensor):
            print("Warning: plot_results expects PyTorch tensors for gts and preds.")
            if self.ax_results_placeholder:
                self.ax_results_placeholder.set_visible(True)
                self.ax_results_placeholder.text(0.5, 0.5, 'Invalid Input',
                                                 ha='center', va='center', fontsize=12)
            return

        batch_size = gts.shape[0]
        if batch_size == 0 or preds.shape[0] == 0:
            print("Warning: gts or preds tensor is empty.")
            if self.ax_results_placeholder:
                self.ax_results_placeholder.set_visible(True)
                self.ax_results_placeholder.text(0.5, 0.5, 'Empty Input',
                                                 ha='center', va='center', fontsize=12)
            return

        # Random sampling
        n_plot = min(self.num_samples_to_plot, batch_size)
        indices = random.sample(range(batch_size), k=n_plot)

        # Determine grid layout
        if grid_layout:
            rows, cols = grid_layout
            if rows * cols < n_plot:
                warnings.warn(f"Grid {grid_layout} too small for {n_plot} samples, adjusting cols.")
                cols = math.ceil(n_plot / rows)
        else:
            rows = 1 if n_plot <= 2 else 2
            cols = math.ceil(n_plot / rows)

        # Setup GridSpec
        gs = self.ax_results_placeholder.get_gridspec()
        sub_gs = gs[0].subgridspec(rows, cols, wspace=0.4, hspace=0.4)
        self.ax_results_placeholder.set_visible(False)

        # Plot each sampled example
        for i, idx in enumerate(indices):
            ax = self.fig.add_subplot(sub_gs[i // cols, i % cols])
            self.results_axes.append(ax)

            gt = gts[idx].detach().cpu().numpy()
            pr = preds[idx].detach().cpu().numpy()

            ax.plot(gt, linestyle='--', marker='.', markersize=4, label='Ground-Truth')
            ax.plot(pr, linestyle='-', marker='.', markersize=4, label='Prediction')
            ax.set_title(f"Sample {idx}")
            ax.tick_params(axis='both', which='major', labelsize=8)
            if i // cols == rows - 1:
                ax.set_xlabel("Dimension", fontsize=9)
            if i % cols == 0:
                ax.set_ylabel("Value", fontsize=9)
            ax.grid(True, linestyle=':', linewidth=0.5)

        try:
            self.fig.tight_layout(pad=2.0)
        except ValueError:
            print("Warning: tight_layout failed. Plots might overlap.")

    def update(self):
        """Redraw canvas."""
        if not self.fig:
            return
        self.fig.canvas.draw_idle()
        plt.pause(0.01)

    def finalize(self):
        """Finalize interactive plot."""
        if not self.fig:
            return
        self.fig.canvas.draw_idle()
        if self._interactive_mode:
            plt.ioff()
            self._interactive_mode = False
        plt.show()

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
