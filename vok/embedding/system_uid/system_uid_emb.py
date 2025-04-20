import numpy as np


class SystemUIDEmbedding:
    def __init__(self):
        self.generator_id = 'system_uid'
        self.dimension = 32

    @classmethod
    def generate_embedding(cls, system_uid, dim=32):
        # Define the modulus for a 64-bit unsigned integer (0 to 2^64 - 1)
        m = 2**64
        
        # Normalize the UID to be within the valid range
        # This handles negative numbers by using modulo arithmetic.
        seed = system_uid % m
        
        # Initialize a new random number generator with the given seed.
        rng = np.random.default_rng(seed)
        
        # Generate a dim-dimensional latent vector with values from a standard normal distribution.
        # You can change to other distributions or apply normalization if needed.
        embedding = rng.standard_normal(size=dim)
        
        # Return the resulting vector.
        return embedding


if __name__ == '__main__':
# Example parameters for an LCG (these are illustrative)
     # offset: number of steps to skip ahead
    gen = SystemUIDEmbedding()
    uid = -123456789012456789+ np.random.randint(0, 1000000000000000000)  # Example UID, which can be negative.
    vector = gen.generate_embedding(uid, dim=4)
    print(vector)