from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import random

from vok.embedding.static_emb.static_emb_utils import StaticEmbeddingUtils

class MultiResolutionSampleCreator:
    def __init__(self, df, sensor_type, source_id, sample_counts, sample_length=144, original_frequency=600):
        """
        Parameters:
          df               : pandas DataFrame with a datetime index and a "value" column (10-minute frequency).
          sensor_type      : Sensor type (string).
          source_id        : Source ID (string).
          sample_counts    : Dictionary mapping downsampling factor to sample count.
                             e.g. {1: 10, 2: 5, 3: 3, 6: 2}
          sample_length    : Number of consecutive rows per sample (default 144).
          original_frequency: Original sampling interval in seconds (default 600 seconds = 10 minutes).
        """
        self.df = df.sort_index()  # Ensure the DataFrame is sorted by datetime.
        self.sensor_type = sensor_type
        self.source_id = source_id
        self.sample_counts = sample_counts
        self.sample_length = sample_length
        self.original_frequency = original_frequency

    def _downsample(self, factor):
        """
        Downsamples the original DataFrame by the given factor.
        For factor==1, returns the original DataFrame.
        For factor > 1, uses resample with a frequency of factor*10 minutes.
        """
        if factor == 1:
            return self.df.copy()
        else:
            freq_str = f'{factor * 10}min'
            downsampled = self.df.resample(freq_str).mean()
            # Drop any potential NaNs (should not occur if the original DF is complete)
            downsampled = downsampled.dropna()
            return downsampled

    def _create_samples_from_df(self, df_down, factor, num_samples):
        """
        From the provided downsampled DataFrame, randomly extracts num_samples samples,
        each consisting of sample_length (default 144) consecutive rows.
        """
        samples = []
        total_rows = len(df_down)
        effective_freq = self.original_frequency * factor  # new frequency in seconds
        
        if total_rows < self.sample_length:
            raise ValueError(f"Downsampled DataFrame (factor {factor}) is too short for the specified sample length.")

        for _ in range(num_samples):
            max_start = total_rows - self.sample_length
            start_idx = random.randint(0, max_start)
            end_idx = start_idx + self.sample_length
            sample_slice = df_down.iloc[start_idx:end_idx]
            data = sample_slice["value"].to_numpy(dtype=np.float32)
            start_ts = sample_slice.index[0].timestamp()
            end_ts = sample_slice.index[-1].timestamp()
            sample_obj = {
                "data": data,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "frequency": effective_freq,
                "sensor_type": self.sensor_type,
                "source_id": self.source_id
            }
            samples.append(sample_obj)
        return samples

    def create_samples(self):
        """
        For each downsampling factor (key) in sample_counts, downsample the DataFrame,
        then randomly pick sample windows (of sample_length rows).
        Returns a combined list of sample objects.
        """
        all_samples = []
        for factor, count in self.sample_counts.items():
            ds_df = self._downsample(factor)
            samples = self._create_samples_from_df(ds_df, factor, count)
            all_samples.extend(samples)
        return all_samples

# ---------------------------
# Example Usage:
# ---------------------------
if __name__ == "__main__":
    # Create a sample DataFrame with a datetime index (every 10 minutes) and a "value" column.
    # rng = pd.date_range(start="2023-01-01", periods=10000, freq="10min")
    # df = pd.DataFrame({"value": np.sin(np.linspace(0, 50, len(rng)))}, index=rng)
    df = pd.read_pickle('notebooks/daily_data.pkl')
    df.fillna(0, inplace=True)
    print(df)
    df.set_index('date', inplace=True)
    # Convert to list of sequences
    df = df[['normalized']]
    df.rename(columns={'normalized': 'value'}, inplace=True)
    print(df)
    # Define how many samples to generate for each resolution.
    # For example, use 10 samples at original resolution, 5 at 20-minute resolution, etc.
    sample_counts = {1: 1, 2: 1, 3: 1, 6: 1}

    # Create the multi-resolution sample creator.
    sample_creator = MultiResolutionSampleCreator(df, sensor_type="temp", source_id="sensor_01", sample_counts=sample_counts)
    
    # Generate samples.
    samples = sample_creator.create_samples()

    # Print a summary for each sample.
    for i, sample in enumerate(samples):
        print(f"Sample {i+1}:")
        print(f"  Data shape: {sample['data'].shape}")
        print(f"  Start ts: {sample['start_ts']}, End ts: {sample['end_ts']}")
        print(f"  Frequency (sec): {sample['frequency']}")
        print(f"  Sensor Type: {sample['sensor_type']}, Source ID: {sample['source_id']}\n")
        print(sample['data'])
        plt.plot(sample['data'])
        embed = StaticEmbeddingUtils.encode_data_context(sample['sensor_type'], sample['source_id'], sample['start_ts'], sample['end_ts'], sample['frequency'])
        print(embed)
    plt.show()
