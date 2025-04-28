

from email.headerregistry import DateHeader
import random
from dawn_vok.vok.embedding.encoders.source_st_enc import TimeStampSTEncoder
from dawn_vok.vok.pipelines.meta_data.raw_data_request.raw_data_retriever_dataset import RawDataRetrieverDataset
from dawn_vok.vok.pipelines.meta_data.time_classifier.time_classefier_dataset import TimeClassifierDataset
from dawn_vok.vok.pipelines.meta_data.time_classifier.time_classefier_model import TimeClassifierModel
from dawn_vok.vok.vmodel.models.basic_models.encoder_decoder import EncoderDecoderModel
from dawn_vok.vok.vmodel.models.v_model import VOKModel
from dawn_vok.vok.vmodel.pipeline.v_pipeline_node import VOKPipelineNode
from dawn_vok.vok.vmodel.trainer.v_trainer import VOKTrainer
from dawn_vok.vok.pipelines.emp.syntax_reducer.emp_syntax_reducer_datasets import EMPSyntaxReducerDataset, EMPSyntaxReducerInferenceDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

class TimeClassifierTrainer(VOKTrainer):
    def __init__(self, node, device, load_model=True, **kwargs):
        super().__init__(node, device, load_model)
        self.criterion = nn.MSELoss()
        self.time_stamp_encoder = TimeStampSTEncoder()
        self.start_year = kwargs.get("start_year", self.time_stamp_encoder.get_config()['min_year'])
        self.end_year = kwargs.get("end_year", self.time_stamp_encoder.get_config()['max_year'])
        self.minute_bin_size = kwargs.get("block_minutes", 1)
        self.input_dim = kwargs.get("input_dim", 16)
        self.loss_weights = {"year":0.3, "month":0.15, "day":0.1, "hour":0.05, "minute":0.05}
        self.heads_to_report_error = ["year", "month", "day", "hour", "minute"]
     
    def run_batch(self, data_batch):
        # unpack the whole batch
        x, y = data_batch

        # if you’re using a device field, move tensors there:
        x = x.to(self.device)
        # y = y.to(self.device)
        logits, latents = self.pipeline_node.model(x)
        # run your pipeline once
        batch_ret = {'preds': logits, 'y': y, 'latents': latents}
        # make sure 'y' (and mask, if needed) are in the returned dict
        # batch_ret['y'] = y
        # batch_ret['mask'] = mask

        return batch_ret
    
    def time_collate_fn(self, batch):
        xs, ts_list = zip(*batch)
        xs = torch.stack(xs, dim=0)
        return xs, list(ts_list)

    def compute_loss(self, batch_ret):
        preds = batch_ret['preds']
        y = batch_ret['y']
        batch_size = len(y)
        total_loss = 0.0
        targets = { k: torch.empty(batch_size, dtype=torch.long, device=self.device) for k in self.loss_weights }

        for i, ts in enumerate(y):
            if "year" in targets:   targets["year"][i]   = ts.year - self.start_year
            if "month" in targets:  targets["month"][i]  = ts.month - 1
            if "day" in targets:    targets["day"][i]    = ts.day - 1
            if "hour" in targets:   targets["hour"][i]   = ts.hour
            if "minute" in targets: targets["minute"][i] = ts.minute // self.minute_bin_size

        loss_components = {}
        for k, target_tensor in targets.items():
            #  preds = preds[0]
             if k not in preds: 
                 print(f"Head '{k}' not in preds")
                 print(f"Preds: {preds.keys()}")
                 continue
             if k not in self.loss_weights: continue
            #  print(f"Target tensor: {target_tensor}")
             n_cls = preds[k].size(1)
             if target_tensor.max() >= n_cls or target_tensor.min() < 0:
                 raise ValueError(f"Head '{k}' target index out of bounds [0, {n_cls-1}]")

             head_loss = F.cross_entropy(preds[k], target_tensor)
             loss_components[k] = head_loss.item()
             total_loss += self.loss_weights[k] * head_loss
        # print(f"Total loss: {total_loss}")
        # print(f"Loss components: {loss_components}")
        # print(f"Preds: {preds}")
        # print(f"Targets: {targets}")
        return total_loss, loss_components
    
    def update_epoch_info(self, batch_ret, epoch_total_errors):
        logits = batch_ret['preds']
        ts_batch = batch_ret['y']
        with torch.no_grad():
            actual_vals = {} # (Calculation as before)
            if "year" in self.heads_to_report_error: actual_vals["year"] = torch.tensor([ts.year for ts in ts_batch], dtype=torch.float32, device=self.device)
            if "month" in self.heads_to_report_error: actual_vals["month"] = torch.tensor([ts.month for ts in ts_batch], dtype=torch.float32, device=self.device)
            if "day" in self.heads_to_report_error: actual_vals["day"] = torch.tensor([ts.day for ts in ts_batch], dtype=torch.float32, device=self.device)
            if "hour" in self.heads_to_report_error: actual_vals["hour"] = torch.tensor([ts.hour for ts in ts_batch], dtype=torch.float32, device=self.device)
            if "minute" in self.heads_to_report_error: actual_vals["minute"] = torch.tensor([ts.minute for ts in ts_batch], dtype=torch.float32, device=self.device)

            for k in self.heads_to_report_error:
                if k not in logits: continue
                pred_indices = torch.argmax(logits[k], dim=1)
                pred_values = None
                if k == "year": pred_values = pred_indices.float() + self.start_year
                elif k == "month": pred_values = pred_indices.float() + 1
                elif k == "day": pred_values = pred_indices.float() + 1
                elif k == "hour": pred_values = pred_indices.float()
                elif k == "minute": pred_values = pred_indices.float() * self.minute_bin_size
                if pred_values is not None:
                    batch_errors = torch.abs(pred_values - actual_vals[k])
                    batch_total_error = torch.sum(batch_errors)
                    epoch_total_errors[k] += batch_total_error.item()
        return epoch_total_errors

    def train(self, train_dataset, val_dataset=None, epochs=10, log_interval=10, lr=None, batch_size=128):
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                        collate_fn=self.time_collate_fn,
                                        num_workers=0)
        if val_dataset is not None:
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        else:
            self.val_loader = None
        
           # Store timestamps from the last batch
        best_loss = float('inf')
        for epoch in range(epochs):
            self.pipeline_node.train()
            total_loss = 0.0
            num_samples = 0
            last_batch_logits = None
            epoch_loss = 0.0
            epoch_loss_components = {k: 0.0 for k in self.loss_weights}
            epoch_total_errors = {k: 0.0 for k in self.heads_to_report_error}
            last_batch_ts = None

            for batch_idx, data_batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                batch_ret = self.run_batch(data_batch)
                if batch_idx == len(self.train_loader) - 1:
                    last_batch_logits = batch_ret['preds']
                    last_batch_ts = batch_ret['y']
                loss, loss_comps = self.compute_loss(batch_ret)
                epoch_total_errors = self.update_epoch_info(batch_ret, epoch_total_errors)
                
                loss.backward()
            
                self.optimizer.step()
                epoch_loss += loss.item() * batch_size
                for comp_k in epoch_loss_components:
                     if comp_k in loss_comps:
                         epoch_loss_components[comp_k] += loss_comps[comp_k] * batch_size
                num_samples += batch_size
                
            if num_samples > 0:
                avg_epoch_loss = epoch_loss / num_samples
                avg_comps = {k: v / num_samples for k, v in epoch_loss_components.items()}
                avg_errors = {k: v / num_samples for k, v in epoch_total_errors.items()}
            else: # Handle empty dataset case
                avg_epoch_loss = 0.0
                avg_comps = {k: 0.0 for k in self.loss_weights}
                avg_errors = {k: 0.0 for k in self.heads_to_report_error}
            if best_loss > avg_epoch_loss:
                best_loss = avg_epoch_loss
                self.pipeline_node.save_model()
            print(f"Epoch {epoch+1}/{epochs}  Avg CE Loss: {avg_epoch_loss:.6f}")
            error_str = "  Avg Abs Errors: "
            unit_map = {"year": "yrs", "month": "mon", "day": "dys", "hour":"hrs", "minute": "min"}
            error_parts = []
            for k in self.heads_to_report_error:
                if k in avg_errors:
                    error_parts.append(f"{k.capitalize()}={avg_errors[k]:.2f} {unit_map.get(k, '')}")
            error_str += " | ".join(error_parts)
            print(error_str)
            if last_batch_logits is not None and last_batch_ts is not None:
                try: # Wrap in try-except in case something goes wrong with indexing/conversion
                    with torch.no_grad():
                        # Use the first sample from the last batch
                        sample_idx = random.randint(0, len(last_batch_ts)-1)
                        original_dt = last_batch_ts[sample_idx]

                        # Get predicted indices for this sample
                        pred_indices = {}
                        for k in self.heads_to_report_error:
                            if k in last_batch_logits:
                                pred_indices[k] = torch.argmax(last_batch_logits[k][sample_idx,:]).item()
                        pred_dt = self.time_stamp_encoder.decode_batch_logits(last_batch_logits, self.minute_bin_size)
                        # Convert indices to values
                        # pred_year = pred_indices.get("year", -1) + self.start_year
                        # pred_month = pred_indices.get("month", -1) + 1
                        # pred_day = pred_indices.get("day", -1) + 1
                        # pred_hour = pred_indices.get("hour", -1)
                        # pred_minute = pred_indices.get("minute", -1) * self.minute_bin_size
                        # pred_dt = datetime(pred_year, pred_month, pred_day, pred_hour, pred_minute)
                        print("  --- Sample Comparison (Last Batch, Idx 0) ---")
                        print(f"  Original DT : {original_dt.strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"  Predicted   : {pred_dt[sample_idx].strftime('%Y-%m-%d %H:%M:%S')}")
                        print("  ---------------------------------------------")
                except Exception as e:
                    print(f"  Error printing sample comparison: {e}") # Print error if sample display fails

                # Clear last batch info for next epoch
            last_batch_logits = None
            last_batch_ts = None
        
        
class TimeClassifierPLN(VOKPipelineNode):
    def __init__(self, node_id, name=None, model_info=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = TimeClassifierModel()
        
        super().__init__(node_id, name, model_info, model=model)
        self.load_trainer()

    def load_trainer(self):
        if self.trainer is None:
            self.trainer = TimeClassifierTrainer(self, 
                             device=self.device, 
                             load_model=True)
        return self.trainer
    
    def train_model(self, dataset=None):
        self.load_trainer()
        dataset = dataset or TimeClassifierDataset(length=50000)
        print('train_model', len(dataset.data))
        self.trainer.train(dataset, epochs=100)

    # def get_latents(self, dataset=None, key=None):
    #     key = key or f'lat_{self.model.latent_dim}'
    #     self.load_trainer()
    #     if dataset is None:
    #         dataset = EMPSyntaxReducerInferenceDataset()
    #     latent_di = self.trainer.inference(dataset, as_dict=True)
    #     dataset.update_latents(latent_di, key=key)
    #     return latent_di


if __name__ == "__main__":
    pln = TimeClassifierPLN(node_id=f'test_time_classifier', name=f'TimeClassifierPLN')
    for i in range(10):
        pln.train_model()
        # print(latents.shape)