import stim
import numpy as np
import h5py
import os

DISTANCE = 3
ROUNDS_LIST = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
NOISE_LEVELS = [0.001, 0.002, 0.003, 0.005, 0.007, 0.01]
SHOTS_PER_CONFIG = 100_000
OUTPUT_DIR = "/scratch/project_465003017/aq1_decoder/data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate(distance, rounds, noise, shots):
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=noise,
        before_measure_flip_probability=noise,
        after_reset_flip_probability=noise,
    )
    sampler = circuit.compile_detector_sampler()
    syndromes, observables = sampler.sample(
        shots=shots,
        separate_observables=True
    )
    return syndromes.astype(np.bool_), observables.astype(np.bool_)

print("Starting data generation...")
total = len(ROUNDS_LIST) * len(NOISE_LEVELS)
done = 0

with h5py.File(f"{OUTPUT_DIR}/d{DISTANCE}_zbasis.h5", "w") as f:
    for rounds in ROUNDS_LIST:
        for noise in NOISE_LEVELS:
            key = f"r{rounds}_p{str(noise).replace('.', '')}"
            syndromes, observables = generate(DISTANCE, rounds, noise, SHOTS_PER_CONFIG)
            grp = f.create_group(key)
            grp.create_dataset("syndromes", data=syndromes)
            grp.create_dataset("observables", data=observables)
            done += 1
            print(f"[{done}/{total}] rounds={rounds} noise={noise} "
                  f"| syndromes shape: {syndromes.shape}")

print(f"Done. Saved to {OUTPUT_DIR}/d{DISTANCE}_zbasis.h5")
