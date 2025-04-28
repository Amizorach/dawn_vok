import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dawn_vok.utils.dict_utils import DictUtils
from dawn_vok.utils.dir_utils import DirUtils
from dawn_vok.vok.v_objects.vok_object import VOKObject
from dawn_vok.vok.vmodel.utils.plotter import Plotter
from torch.utils.data import DataLoader
import torch.nn.functional as F
class VOKTrainer(VOKObject):
    def __init__(self, pipeline_node, device=None,
                 criterion=None,
                 optimizer=None,
                 lr=None,
                 filename='best_model1.pth', 
                 path=None,
                 load_model=True):
        self.filename = filename or 'best_model1.pth'
        self.pipeline_node = pipeline_node
        self.device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pipeline_node.to(self.device)
        self.lr = lr or 0.0001
        self.criterion = nn.MSELoss()#criterion or self.pipeline_node.get_criterion()
        self.optimizer = optim.Adam(self.pipeline_node.model.parameters(), lr=self.lr)#optimizer or self.pipeline_node.get_optimizer()

        if load_model:
            self.pipeline_node.load_model()

        self.loss_history = []
        self.plotter = Plotter(plot_loss_axis=True,
                               plot_results_axis=True) # Using old flag name

    def compute_masked_loss(self, reconstruction, target, mask):
        """
        Computes the masked mean squared error loss.
        
        Args:
            reconstruction (torch.Tensor): Reconstructed output of shape [B, T, D].
            target (torch.Tensor): Ground truth tensor of shape [B, T, D].
            mask (torch.Tensor): Mask tensor of shape [B, T], where 1 indicates valid positions.
            
        Returns:
            torch.Tensor: The computed masked loss.
        """
        # Expand mask to match the dimensions of the reconstruction/target.
        # mask_expanded shape: [B, T, 1]
        mask_expanded = mask.unsqueeze(-1)
        
        # Compute the element-wise squared error.
        squared_error = (reconstruction - target) ** 2
        
        # Apply the mask: Only valid positions contribute to the loss.
        masked_squared_error = squared_error * mask_expanded
        
        # Sum the errors and then divide by the number of valid elements.
        # The total number of valid elements is the sum of mask values multiplied by the feature dimension.
        total_valid = mask_expanded.sum()
        
        # Avoid division by zero.
        if total_valid == 0:
            return torch.tensor(0.0, device=reconstruction.device)
        
        loss = masked_squared_error.sum() / total_valid
        return loss
    
    def compute_masked_loss_cosine(self, reconstruction, target, mask):
        """
        Computes the masked cosine‐distance loss.

        Args:
            reconstruction (torch.Tensor): [B, T, D] reconstructed output.
            target         (torch.Tensor): [B, T, D] ground‑truth.
            mask           (torch.Tensor): [B, T] binary mask (1 = valid, 0 = ignore).

        Returns:
            torch.Tensor: Scalar masked cosine loss.
        """
        # Compute cosine similarity per time‑step: [B, T]
        cos_sim = F.cosine_similarity(reconstruction, target, dim=-1)

        # Convert similarity to distance: (1 - cos_sim)
        cos_dist = 1.0 - cos_sim

        # Mask out invalid positions
        masked_dist = cos_dist * mask

        # Count valid positions
        total_valid = mask.sum()

        # Avoid div by zero
        if total_valid.item() == 0:
            return torch.tensor(0.0, device=reconstruction.device)

        # Average over all valid entries
        loss = masked_dist.sum() / total_valid * 100
        return loss
   
    def run_batch(self, data_batch):
        # unpack the whole batch
        x, y, mask = data_batch

        # if you’re using a device field, move tensors there:
        x = x.to(self.device)
        y = y.to(self.device)
        mask = mask.to(self.device)

        # run your pipeline once
        reconstructed, encoded = self.pipeline_node.model(x)
        batch_ret = {'preds': reconstructed, 'encoded': encoded, 'y': y, 'mask': mask}
        # make sure 'y' (and mask, if needed) are in the returned dict
        # batch_ret['y'] = y
        # batch_ret['mask'] = mask

        return batch_ret
    
    def compute_loss(self, batch_ret):
        preds = batch_ret['preds']
        y = batch_ret['y']
        mask = batch_ret.get('mask', None)
        loss = self.compute_masked_loss(preds, y, mask)
        return loss
    
    def train(self, train_dataset, val_dataset=None, epochs=1000, log_interval=10, lr=None, batch_size=128):
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if val_dataset is not None:
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        else:
            self.val_loader = None
        best_loss = float('inf')

        for epoch in range(1, epochs+1):
            self.pipeline_node.train()
            total_loss = 0.0
            num_samples = 0
            last_batch_ret = None # Store the dictionary from the last successful batch

            for batch_idx, data_batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                batch_ret = self.run_batch(data_batch)
                loss = self.compute_loss(batch_ret)
              
                # l oss_cosine = self.compute_masked_loss_cosine(preds, y, mask)
                loss.backward()
                try:
                    current_batch_size = len(batch_ret)
                    num_samples += current_batch_size
                    total_loss += loss.detach().item() * current_batch_size
                except KeyError:
                    print(f"Warning: 'y' key missing in batch_ret during validation.")

               
                self.optimizer.step()
                last_batch_ret = batch_ret

            avg_loss = total_loss / len(self.train_loader.dataset)
            self.loss_history.append(avg_loss)
            print(f"Epoch {epoch}/{epochs} - Train MSE: {avg_loss:.6f} - best: {best_loss:.6f} - lr: {self.optimizer.param_groups[0]['lr']}")
            if avg_loss < best_loss:


                best_loss = avg_loss
                self.pipeline_node.save_model(self.filename)
            last_batch_ret = None
            if epoch % log_interval == 0:
                # Always plot loss if axis exists
                self.plotter.plot_loss(self.loss_history)

                if last_batch_ret is not None:
                    # print(f"last_batch_ret: {last_batch_ret['y'].keys()}")
                    y_batch = last_batch_ret['y']['start']
                    batch_size = len(last_batch_ret['preds'])

                    # choose one sample
                    random_index = random.randrange(batch_size)

                        # sum over all dims except batch‑dim, then .item()
                    # sumy = y_batch[random_index].sum().item()
                    # print('sumy',sumy)
                    # if sumy != 0:
                    preds = last_batch_ret['preds']['start']
                    if preds.ndim == 1:
                        preds = preds.reshape(1, -1)
                    # print(preds.shape)
                    # print(last_batch_ret['y'][random_index].shape)
                    # exit()
                    # print(f"y_batch[random_index]: {y_batch[random_index]}")
                    # print(f"preds: {preds}")
                    self.plotter.plot_results(y_batch, preds)
                    # else:
                    # # This case happens if the epoch had no successful batches
                    #     print("  Skipping sample plotting (no successful batch processed in epoch).")

            # Update the plot window regardless of sample plotting success
            self.plotter.update()
        # --- End Plotting ---

        # --- Validation Phase ---
            if self.val_loader is not None:
                self.validate(epoch) # validate method uses run_batch
              
        print("Training finished.")
        self.plotter.finalize() # Finalize plot after all epochs  and close the plot without waiting for user input
        # plt.close()
        self.pipeline_node.save_model(self.filename)
    

        

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
     
    def run_batch_inference(self, data_batch):
        # unpack the whole batch
        x, meta_data = data_batch

        # if you’re using a device field, move tensors there:
        x = x.to(self.device)

        # run your pipeline once
        reconstructed, encoded = self.pipeline_node.model(x)
        batch_ret = {'preds': reconstructed, 'encoded': encoded, 'y': x, 'meta_data': meta_data}
        # make sure 'y' (and mask, if needed) are in the returned dict
        # batch_ret['y'] = y
        # batch_ret['mask'] = mask

        return batch_ret
    
    def finalize_inference(self, latents, dataset):
        batch_ret = {}
        for i, latent in enumerate(latents):
            batch_ret[dataset.meta_data[i]['latent_map']] = latent
        return batch_ret
    
    def inference(self, dataset, as_dict=False, key='encoded'):

        self.pipeline_node.eval()
        print('dataset', len(dataset))
        self.inference_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        latents = []
        for batch_idx, data_batch in enumerate(self.inference_loader):
            batch_ret = self.run_batch_inference(data_batch)
            if key in batch_ret:
                latents.append(batch_ret[key])
            else:
                print(f"Key '{key}' not found in batch_ret for batch {batch_ret.keys()}")

        latents = torch.cat(latents, dim=0)
        if as_dict:
            return self.finalize_inference(latents, dataset)
        else:
            return latents
if __name__ == '__main__':
  
  pass