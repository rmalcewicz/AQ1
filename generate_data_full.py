import stim
import numpy as np
import h5py
import os

DISTANCE = 3
ROUNDS_LIST = list(range(1, 21))  # 1 through 20
NOISE_LEVELS = [0.001, 0.002, 0.003, 0.005, 0.007, 0.01]
SHOTS_PER_CONFIG = 500_000
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "data")

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
    syndromes, observables = sampler.sample(shots=shots, separate_observables=True)
    return syndromes.astype(np.bool_), observables.astype(np.bool_)

print("Starting FULL Tier 1 data generation...")
print(f"Total configs: {len(ROUNDS_LIST)} rounds x {len(NOISE_LEVELS)} noise = {len(ROUNDS_LIST)*len(NOISE_LEVELS)}")
print(f"Total samples: {len(ROUNDS_LIST)*len(NOISE_LEVELS)*SHOTS_PER_CONFIG:,}")
total = len(ROUNDS_LIST) * len(NOISE_LEVELS)
done = 0

with h5py.File(f"{OUTPUT_DIR}/d3_zbasis_full.h5", "w") as f:
    f.attrs['distance'] = DISTANCE
    f.attrs['basis'] = 'Z'
    f.attrs['detector_convention'] = 'stim_compile_detector_sampler'
    for rounds in ROUNDS_LIST:
        for noise in NOISE_LEVELS:
            key = f"r{rounds}_p{str(noise).replace('.', '')}"
            syndromes, observables = generate(DISTANCE, rounds, noise, SHOTS_PER_CONFIG)
            grp = f.create_group(key)
            grp.create_dataset("syndromes", data=syndromes)
            grp.create_dataset("observables", data=observables)
            grp.attrs['rounds'] = rounds
            grp.attrs['noise'] = noise
            grp.attrs['shots'] = SHOTS_PER_CONFIG
            done += 1
            print(f"[{done}/{total}] rounds={rounds} noise={noise:.3f} "
                  f"| syndromes {syndromes.shape} obs {observables.shape}")

print(f"Done. Saved to {OUTPUT_DIR}/d3_zbasis_full.h5")
