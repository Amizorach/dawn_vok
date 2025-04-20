import numpy as np

class LatentGridInterpreter:
    def __init__(self):
        # Define known mappings (example)
        self.source_ids = {0: "source_A", 1: "source_B", 2: "source_C"}
        self.sensor_types = {0: "tension", 1: "humidity", 2: "radiation"}
        self.agg_methods = {0: "mean", 1: "min", 2: "max"}

        # Each slot has (slice range, handler)
        self.slots = {
            "source_id": ((0, 4), self._categorical_lookup(self.source_ids)),
            "sensor_type": ((4, 8), self._categorical_lookup(self.sensor_types)),
            "time_window": ((8, 12), self._continuous_range(min_val=6, max_val=72)),
            "agg_type": ((12, 16), self._categorical_lookup(self.agg_methods)),
            "start_time_offset": ((16, 20), self._continuous_range(0, 10000)),  # e.g., hours from base
        }

    def _categorical_lookup(self, id_map):
        def fn(vec):
            idx = int(np.argmax(vec)) % len(id_map)
            return id_map[idx]
        return fn

    def _continuous_range(self, min_val, max_val):
        def fn(vec):
            val = np.tanh(np.mean(vec))  # squash to [-1,1]
            val = (val + 1) / 2  # scale to [0,1]
            return min_val + val * (max_val - min_val)
        return fn

    def interpret(self, latent, keys=None):
        result = {}
        keys = keys or self.slots.keys()

        for key in keys:
            idx_range, handler = self.slots[key]
            result[key] = handler(latent[idx_range[0]:idx_range[1]])

        return result

if __name__ == "__main__":
    interpreter = LatentGridInterpreter()
    latent = np.random.rand(20)
    print(latent)
    print(interpreter.interpret(latent))
