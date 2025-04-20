import numpy as np

def test_generation_consistency(seed=42, dimension=16, n_small=3, n_large=30):
    """
    Tests if generating standard normal vectors with a fixed seed produces
    identical initial vectors regardless of the total number generated in the batch.

    Args:
        seed (int): The fixed seed for the random number generator.
        dimension (int): The dimension of the vectors.
        n_small (int): The number of vectors to generate in the first (small) batch.
        n_large (int): The number of vectors to generate in the second (large) batch.
                       Must be >= n_small.
    """
    if not isinstance(seed, int): raise TypeError("Seed must be an integer.")
    if not isinstance(dimension, int) or dimension <= 0: raise ValueError("Dimension must be positive integer.")
    if not isinstance(n_small, int) or n_small <= 0: raise ValueError("n_small must be positive integer.")
    if not isinstance(n_large, int) or n_large < n_small: raise ValueError("n_large must be >= n_small.")

    print(f"--- Testing Generation Consistency ---")
    print(f"Seed: {seed}, Dimension: {dimension}")
    print(f"Comparing first vector from generating {n_small} vs {n_large} vectors.")

    # --- Generation 1 (Small Batch) ---
    try:
        rng1 = np.random.Generator(np.random.PCG64(seed))
        mat_small = rng1.standard_normal(size=(dimension, n_small), dtype=np.float32)
        # Get the first vector (index 0)
        vector_from_small = mat_small[:, 0]
        print(f"\nGenerated small matrix ({dimension}, {n_small}).")
        print(f"Vector at index 0 (sample): {vector_from_small[:5]}...")
    except Exception as e:
        print(f"Error during small batch generation: {e}")
        return

    # --- Generation 2 (Large Batch) ---
    try:
        # Re-initialize generator with the SAME seed
        rng2 = np.random.Generator(np.random.PCG64(seed))
        mat_large = rng2.standard_normal(size=(dimension, n_large), dtype=np.float32)
        # Get the first vector (index 0)
        vector_from_large = mat_large[:, 0]
        print(f"\nGenerated large matrix ({dimension}, {n_large}).")
        print(f"Vector at index 0 (sample): {vector_from_large[:5]}...")
    except Exception as e:
        print(f"Error during large batch generation: {e}")
        return

    # --- Comparison ---
    print("\n--- Comparison ---")
    # Using allclose for robustness with floating-point numbers
    are_close = np.allclose(vector_from_small, vector_from_large)
    print(f"Are vectors for index 0 close (np.allclose)? : {are_close}")

    # Also check strict equality
    are_equal = np.array_equal(vector_from_small, vector_from_large)
    print(f"Are vectors for index 0 equal (np.array_equal)? : {are_equal}")

    if not are_close:
        print("\n*** WARNING: Vectors differ significantly! Deterministic generation failed. ***")
    elif not are_equal:
        print("\nNote: Vectors are close but not bit-for-bit equal (minor float differences).")
    else:
        print("\nSuccess: Vectors are identical. Deterministic generation confirmed.")

    print("-" * 30)

import numpy as np
import time

SEED = 42
DIMENSION = 16
NUM_VECTORS = 1_000_000 # Start with 1 million first? Or go straight to 10M?

print(f"Timing per-index generation for {NUM_VECTORS} vectors...")
start_time = time.time()

vectors_list = []
for index in range(NUM_VECTORS):
    bit_generator = np.random.PCG64(SEED)
    offset = index * DIMENSION
    if offset > 0:
        try: bit_generator.advance(offset)
        except Exception as e: raise RuntimeError(f"...") from e
    rng = np.random.Generator(bit_generator)
    vector = rng.standard_normal(size=DIMENSION, dtype=np.float32)
    vectors_list.append(vector)
# cpu_matrix = np.stack(vectors_list, axis=-1) # Stacking also takes time

end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds.")
exit()
# Compare to block generation (optional)
# print(f"\nTiming block generation for {NUM_VECTORS} vectors...")
# start_time_block = time.time()
# rng_block = np.random.Generator(np.random.PCG64(SEED))
# cpu_matrix_block = rng_block.standard_normal(size=(DIMENSION, NUM_VECTORS), dtype=np.float32)
# end_time_block = time.time()
# print(f"Block time taken: {end_time_block - start_time_block:.2f} seconds.")
# --- Run the test ---
if __name__ == "__main__":
    # Use dimension=16 and test generating 3 vs 30 vectors
    test_generation_consistency(seed=42, dimension=16, n_small=3, n_large=30)

    # Test with the specific indices from the failing assertion (Index 1)
    # Generate first 5, get index 1
    seed_test = 42 # Use same seed as test
    dim_test = 16
    rng_t1 = np.random.Generator(np.random.PCG64(seed_test))
    mat_t1 = rng_t1.standard_normal(size=(dim_test, 6), dtype=np.float32)
    vec1 = mat_t1[:, 1]

    # Generate first 6, get index 1
    rng_t2 = np.random.Generator(np.random.PCG64(seed_test))
    mat_t2 = rng_t2.standard_normal(size=(dim_test, 6), dtype=np.float32)
    vec2 = mat_t2[:, 1]

    print("\n--- Specific Test for Index 1 (Size 5 vs 6) ---")
    print(f"Vector Index 1 from Size 5 (sample): {vec1[:5]}...")
    print(f"Vector Index 1 from Size 6 (sample): {vec2[:5]}...")
    print(f"np.allclose: {np.allclose(vec1, vec2)}")
    print(f"np.array_equal: {np.array_equal(vec1, vec2)}")
    print("-" * 30)