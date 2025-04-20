import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dawn_vok.utils.dir_utils import DirUtils
from dawn_vok.vok.model_utils.trainer.plotter import Plotter


class BaseTrainer:
    def __init__(self, model, device=None,
                 criterion=None,
                 optimizer=None,
                 filename='best_model1.pth', 
                 path=None,
                 load_model=True):
        self.filename = filename or 'best_model1.pth'
        self.path = path or DirUtils.get_model_path(model_id=model.model_id, version=model.version, path=model.name)
        self.model = model
        self.device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        self.criterion = criterion or model.get_criterion()
        self.optimizer = optimizer or model.get_optimizer()

        if load_model:
            self.model.load_model(self.filename, self.path)

        self.loss_history = []
        self.plotter = Plotter(plot_loss_axis=True,
                               plot_results_axis=True) # Using old flag name


    def train(self, train_loader, val_loader=None, epochs=1000, log_interval=10, lr=None):
              
        self.train_loader = train_loader
        self.val_loader = val_loader
        best_loss = float('inf')
      
        for epoch in range(1, epochs+1):
            self.model.train()
            total_loss = 0.0
            num_samples = 0
            last_batch_ret = None # Store the dictionary from the last successful batch

            for batch_idx, data_batch in enumerate(self.train_loader):
                batch_ret = self.run_batch(data_batch)
        
                self.optimizer.zero_grad()
                if batch_ret is None: # Check if run_batch failed
                    print(f"Skipping invalid validation batch {batch_idx}")
                    continue

                # Calculate loss using the custom criterion
                loss = self.criterion(batch_ret)
                loss.backward()
                try:
                    y = batch_ret['y']
                    current_batch_size = y.size(0)
                    num_samples += current_batch_size
                    total_loss += loss.item() * current_batch_size
                except KeyError:
                    print(f"Warning: 'y' key missing in batch_ret during validation.")

               
                self.optimizer.step()
          

            avg_loss = total_loss / len(self.train_loader.dataset)
            self.loss_history.append(avg_loss)

            print(f"Epoch {epoch}/{epochs} - Train MSE: {avg_loss:.6f} - best: {best_loss:.6f} - lr: {self.optimizer.param_groups[0]['lr']}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.model.save_model(self.filename)
            if epoch % log_interval == 0:
                # Always plot loss if axis exists
                self.plotter.plot_loss(self.loss_history)

                if last_batch_ret is not None:
                    self.plotter.plot_results(last_batch_ret['y'], last_batch_ret['preds'])
                else:
                    # This case happens if the epoch had no successful batches
                    print("  Skipping sample plotting (no successful batch processed in epoch).")

            # Update the plot window regardless of sample plotting success
            self.plotter.update()
        # --- End Plotting ---

        # --- Validation Phase ---
            if self.val_loader is not None:
                self.validate(epoch) # validate method uses run_batch
              
        print("Training finished.")
        self.plotter.finalize() # Finalize plot after all epochs
        

    def validate(self, epoch):
        """Performs validation using self.val_loader and run_batch."""
        self.model.eval() # Ensure model is in evaluation mode
        total_loss = 0.0
        num_samples = 0 # Initialize sample counter for accurate averaging
        with torch.no_grad(): # No gradients needed for validation
            for batch_idx, data_batch in enumerate(self.val_loader):
                # --- Use run_batch ---
                # run_batch handles data parsing, device moving, and forward pass,
                # returning a dictionary {'preds': ..., 'y': ...} or None.
                batch_ret = self.run_batch(data_batch)
                # ---------------------

                # Check if run_batch was successful
                if batch_ret is None:
                    print(f"Skipping invalid validation batch {batch_idx}")
                    continue # Skip to next batch

                # --- Calculate Loss ---
                # Use the custom criterion which expects the dictionary
                try:
                    loss = self.criterion(batch_ret)
                except Exception as e:
                    print(f"Error during validation criterion calculation for batch {batch_idx}: {e}")
                    continue # Skip batch if loss calculation fails
                # --------------------

                # --- Accumulate Loss and Samples ---
                try:
                    # Extract target 'y' to get batch size for accurate avg loss
                    y = batch_ret['y']
                    current_batch_size = y.size(0)

                    num_samples += current_batch_size
                    total_loss += loss.item() * current_batch_size # Accumulate loss weighted by batch size
                except KeyError:
                    # This might happen if run_batch returns dict without 'y'
                    print(f"Warning: 'y' key missing in batch_ret during validation batch {batch_idx}.")
                    # Cannot accurately track samples/average loss if 'y' is missing.
                except Exception as e:
                    print(f"Error during validation loss accumulation: {e}")
                # -----------------------------------

        # --- Calculate Average Loss ---
        # Calculate average using accumulated samples, guarding against division by zero
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        # --------------------------

        # Print result (using a slightly more generic label)
        print(f"Epoch {epoch} - Val   Loss: {avg_loss:.6f}")
     


if __name__ == '__main__':
    from dawn_vok.vok.model_utils.models.vok_model import VSensorTypeLatentModel
    from dawn_vok.vok.model_utils.models.vok_optimizer import VOptimizer
    model = VSensorTypeLatentModel(input_dim=10, hidden_dims=[100, 100], num_types=10, latent_dim=16)
    print(model)
    trainer = BaseTrainer(model)
    trainer.train()