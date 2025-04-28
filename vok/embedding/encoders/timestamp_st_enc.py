from dawn_vok.vok.embedding.encoders.base.model_st_encoder import ModelSTEncoder
from dawn_vok.utils.dict_utils import DictUtils
from datetime import datetime
import torch


class TimeStampSTEncoder(ModelSTEncoder):

    @classmethod
    def get_config(cls):
        return {
            "min_year": 1990,
            "max_year": 2029,
            "info_map": {
                0: "year_norm",
                1: "month_norm",
                2: "month_sin",
                3: "month_cos",
                4: "day_norm",
                5: "day_sin",
                6: "day_cos",
                7: "hour_norm",
                8: "hour_sin",
                9: "hour_cos",
                10: "minute_norm",
                11: "minute_sin",
                12: "minute_cos",
                13: "since_start_norm",
                14: "since_start_sin",
                15: "since_start_cos",
            }
        }
    
    def __init__(self):
        from dawn_vok.vok.pipelines.meta_data.time_classifier.time_classefier_model import TimeClassifierModel

        uid = "time_stamp_encoder"  
        model = TimeClassifierModel()
        super().__init__(uid=uid, model=model)
        self.min_year = self.config["min_year"]
        self.max_year = self.config["max_year"]
        self.year_range = self.max_year - self.min_year
        self.tot_days = (self.max_year - self.min_year) * 365 + (self.max_year - self.min_year) // 4
        self.tot_hours = self.tot_days * 24
        self.tot_minutes = self.tot_hours * 60
        self.dim_size = len(self.config["info_map"])

    def encode(self, dt):
        if dt is not None:
            dt = DictUtils.parse_datetime_direct(dt)
        ts= datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute)
        # 2) Build a feature vector 'x' with informative features
        x = []
        # --- Feature Engineering ---
        # Feature 0: Normalized Year
        # Scale year to be roughly between 0 and 1
        norm_year = (ts.year - self.min_year) / self.year_range
        # Already converting to tensor here, this is fine:
        x.append(torch.tensor(norm_year, dtype=torch.float32))
        # Features 1, 2: Cyclical Month (0-11)
        # Using ts.month - 1 to get a 0-based index for the 12 months
        month_norm = (ts.month - 1) / 12.0 
        month_angle_float = 2 * torch.pi * month_norm

        # Convert float angle to tensor BEFORE sin/cos
        month_angle_tensor = torch.tensor(month_angle_float, dtype=torch.float32)

        x.append(torch.tensor(month_norm, dtype=torch.float32))
        x.append(torch.sin(month_angle_tensor))
        x.append(torch.cos(month_angle_tensor))
        # Features 3, 4: Cyclical Day of Month (0-30 approx)
        # Using ts.day - 1 for 0-based index. Normalizing by 31 is an approximation.
        day_norm = (ts.day - 1) / 31.0
        day_angle_float = 2 * torch.pi * day_norm
        # Convert float angle to tensor BEFORE sin/cos
        day_angle_tensor = torch.tensor(day_angle_float, dtype=torch.float32)
        x.append(torch.tensor(day_norm, dtype=torch.float32))
        x.append(torch.sin(day_angle_tensor))
        x.append(torch.cos(day_angle_tensor))
        # Features 5, 6: Cyclical Hour (0-23)
        hour_norm = ts.hour / 24.0
        hour_angle_float = 2 * torch.pi * hour_norm
        # Convert float angle to tensor BEFORE sin/cos
        hour_angle_tensor = torch.tensor(hour_angle_float, dtype=torch.float32)
        x.append(torch.tensor(hour_norm, dtype=torch.float32))
        x.append(torch.sin(hour_angle_tensor))
        x.append(torch.cos(hour_angle_tensor))
        # Features 7, 8: Cyclical Minute (0-59)

        # Features 7, 8: Cyclical Minute (0-59)
        minute_norm = ts.minute / 60.0
        minute_angle_float = 2 * torch.pi * minute_norm
        # Convert float angle to tensor BEFORE sin/cos
        minute_angle_tensor = torch.tensor(minute_angle_float, dtype=torch.float32)
        x.append(torch.tensor(minute_norm, dtype=torch.float32))
        x.append(torch.sin(minute_angle_tensor))
        x.append(torch.cos(minute_angle_tensor))

        tot_days_since_start = (ts.year - self.min_year) * 365 + (ts.year - self.min_year) // 4 + (ts.month - 1) * 31 + ts.day - 1
        tot_minutes_since_start = tot_days_since_start * 24 * 60 + ts.hour * 60 + ts.minute
        norm_minutes_since_start = tot_minutes_since_start / self.tot_minutes
        total_minutes_angle = torch.tensor(2 * torch.pi * norm_minutes_since_start, dtype=torch.float32)
        x.append(torch.tensor(norm_minutes_since_start, dtype=torch.float32))
        x.append(torch.sin(total_minutes_angle))
        x.append(torch.cos(total_minutes_angle))
        
        v = torch.stack(x, dim=0)
        return v
        
    def decode(self, v):
        v = {self.info_map[i]: float(v[i]) for i in range(len(v))}
        y = v["year_norm"] * self.year_range + self.min_year
        m = v["month_norm"] * 12 + 1
        d = v["day_norm"] * 31 + 1
        h = v["hour_norm"] * 24
        mi = v["minute_norm"] * 60
        try:
            dt = datetime(round(y), round(m), round(d), round(h), round(mi))
        except ValueError:
            print('v', v)
            print('y', y)
            print('m', m)
            print('d', d)
            print('h', h)
            print('mi', mi)
            return None
        return dt
    
    def decode_batch_logits(self, logits: dict, minute_bin_size: int):
        """
        logits: dict of tensors, each of shape [B, num_classes] for heads
        minute_bin_size: the bin size used when training (so we can invert the minute bin)
        Returns: list of datetime objects of length B
        """
        # 1) pick predicted class indices for each head
        preds = { head: torch.argmax(logits[head], dim=1).cpu().numpy()
                  for head in ["year","month","day","hour","minute"]
                  if head in logits }
        #  print(f"Preds: {preds}")
        B = next(iter(preds.values())).shape[0]
        out = []
        for i in range(B):
            # 2) build the “normalized feature” vector v of length len(info_map)
            v = torch.zeros(len(self.info_map), dtype=torch.float32)
            # year_norm lives at index where info_map==“year_norm” (should be 0)
            v[0] = (preds["year"][i]) / self.year_range
            # month_norm at index 1
            v[1] = (preds["month"][i]) / 12.0
            # day_norm at index 4
            v[4] = (preds["day"][i]) / 31.0
            # hour_norm at index 7
            v[7] = (preds["hour"][i]) / 24.0
            # minute_norm at index 10
            v[10] = (preds["minute"][i] * minute_bin_size) / 60.0

            # 3) decode single vector
            dt = self.decode(v)
            out.append(dt)
        
        return out
    
    def decode_from_latent(self, latents):
        lats = self.model.decoder(latents)
        return self.decode_batch_logits(lats, 1)