import os
import torch

from dawn_vok.utils.dir_utils import DirUtils

class EarlyStopping:
    def __init__(self, model_id, version, min_epochs=10, patience=10, min_delta=0.0001, verbose=True, checkpoint_file='best_retrieval_model.pt'):
        """
        Args:
            patience (int): How many epochs to wait before stopping if loss doesn't improve.
            min_delta (float): The minimum change to qualify as an improvement.
            verbose (bool): Whether to print messages.
            path (str): Path to save the best model.
        """
        self.min_epochs = min_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = DirUtils.get_checkpoints_dir(model_id, version, checkpoint_file) if model_id and version and checkpoint_file else None 
        self.best_loss = None
        self.best_epoch = 0
        self.epochs_without_improvement = 0

    def should_stop(self, current_loss, current_epoch, model):
        """
        Checks whether the training should stop based on the current loss and epoch.
        Saves the model if the loss improves.

        Returns:
            bool: Whether to stop training early.
        """
        if self.best_loss is None:
            self.best_loss = current_loss
            self.best_epoch = current_epoch
            self.save_checkpoint(current_epoch, current_loss, model)
            return False

        # Check if loss has improved
        if current_loss < self.best_loss - self.min_delta:
            if self.verbose:
                print(f"Loss improved ({self.best_loss:.6f} --> {current_loss:.6f}). Saving model...")
            self.best_loss = current_loss
            self.best_epoch = current_epoch
            self.epochs_without_improvement = 0
            if self.path:
                self.save_checkpoint(current_epoch, current_loss, model)
            return False
        else:
            if current_epoch < self.min_epochs:
                return False
            self.epochs_without_improvement += 1
            if self.verbose:
                print(f"No improvement for {self.epochs_without_improvement} epochs. current loss: {current_loss:.6f}, best loss: {self.best_loss:.6f}")
            if self.epochs_without_improvement >= self.patience:
                if self.verbose:
                    print(f"Early stopping triggered at epoch {current_epoch + 1} (best loss {self.best_loss:.6f} at epoch {self.best_epoch + 1})")
                return True
            return False

    def save_checkpoint(self, epoch, loss, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.path)
        self.best_loss = loss
        self.best_epoch = epoch
        if self.verbose:
            print(f"Model saved to {self.path}")