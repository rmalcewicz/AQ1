
"""

Generate d=3 surface code training data using REAL IQM Emerald calibration.

Builds a Stim rotated_memory_z circuit with per-qubit PRX/readout errors and

per-pair CZ errors mapped from the live Emerald quality metric set.

"""

import sys

sys.path.insert(0, '/scratch/project_465003017/aq1_decoder/packages')



import json

import numpy as np

import stim

import h5py



METRICS_PATH = "/scratch/project_465003017/aq1_decoder/data/emerald_metrics.json"

OUTPUT_DIR   = "/scratch/project_465003017/aq1_decoder/data"



# ─── Your optimized placement ───

# data qubits at Stim coords (rotated d=3, data on (odd,odd)/(even,even) interior)

DATA = {

    "QB6": (1,1), "QB13": (3,1), "QB22": (5,1),

    "QB11": (1,3), "QB20": (3,3), "QB29": (5,3),

    "QB18": (1,5), "QB27": (3,5), "QB36": (5,5),

}

# X-type ancillas (measure X-stabilizers)

X_ANC = {"QB7": (2,0), "QB21": (4,2), "QB19": (2,4), "QB35": (4,6)}

# Z-type ancillas (measure Z-stabilizers)

Z_ANC = {"QB12": (2,2), "QB30": (6,2), "QB10": (0,4), "QB28": (4,4)}



# Stabilizer -> the data qubits it checks (from coupler topology)

STABILIZERS = {

    "QB7":  ["QB6", "QB13"],                  # X boundary

    "QB21": ["QB13", "QB20", "QB22", "QB29"], # X bulk

    "QB19": ["QB20", "QB11", "QB18", "QB27"], # X bulk

    "QB35": ["QB27", "QB36"],                 # X boundary

    "QB12": ["QB6", "QB13", "QB20", "QB11"],  # Z bulk

    "QB30": ["QB22", "QB29"],                 # Z boundary

    "QB10": ["QB11", "QB18"],                 # Z boundary

    "QB28": ["QB20", "QB29", "QB27", "QB36"], # Z bulk

}



def load_metrics(path):

    with open(path) as f:

        return json.load(f)



def prx_err(m, qb):

    k = f"metrics.rb.prx.drag_crf_sx.{qb}.fidelity:par=d2"

    return max(0.0, 1.0 - m[k]) if k in m else 0.0005



def readout_err(m, qb):

    e01 = m.get(f"metrics.ssro.measure.constant.{qb}.error_0_to_1", 0.01)

    e10 = m.get(f"metrics.ssro.measure.constant.{qb}.error_1_to_0", 0.01)

    return e01, e10



def cz_err(m, qa, qb):

    for a, b in [(qa, qb), (qb, qa)]:

        k = f"metrics.irb.cz.crf_crf.{a}__{b}.fidelity:par=d2"

        if k in m:

            return max(0.0, 1.0 - m[k])

    return 0.005  # fallback



def build_circuit(rounds, m):

    """

    Build a d=3 rotated surface code Z-memory circuit with per-component noise.

    Z-basis memory: prepare |0>_L, measure logical Z.

    Detectors on Z-stabilizers track bit-flips across rounds.

    """

    c = stim.Circuit()



    all_data = list(DATA.keys())

    all_anc  = list(X_ANC.keys()) + list(Z_ANC.keys())

    # assign integer indices

    idx = {q: i for i, q in enumerate(all_data + all_anc)}



    # set coordinates for detectors/visualization

    for q, (x, y) in {**DATA, **X_ANC, **Z_ANC}.items():

        c.append("QUBIT_COORDS", [idx[q]], [x, y])



    data_ids = [idx[q] for q in all_data]

    z_anc_ids = [idx[q] for q in Z_ANC]

    x_anc_ids = [idx[q] for q in X_ANC]



    # ── Initialize: reset all qubits (|0>) ──

    c.append("R", data_ids + z_anc_ids + x_anc_ids)

    for q in all_data:

        e01, _ = readout_err(m, q)

        c.append("X_ERROR", [idx[q]], e01)  # reset error proxy



    def syndrome_round(first=False):

        # X-ancillas get Hadamard to measure X-stabs

        for q in X_ANC:

            c.append("H", [idx[q]])

            c.append("DEPOLARIZE1", [idx[q]], prx_err(m, q))

        # CZ interactions: each ancilla with its data qubits

        for anc, datas in STABILIZERS.items():

            for d in datas:

                c.append("CZ", [idx[anc], idx[d]])

                c.append("DEPOLARIZE2", [idx[anc], idx[d]], cz_err(m, anc, d))

        for q in X_ANC:

            c.append("H", [idx[q]])

            c.append("DEPOLARIZE1", [idx[q]], prx_err(m, q))

        # measure + reset all ancillas with asymmetric readout noise

        for q in all_anc:

            e01, e10 = readout_err(m, q)

            c.append("X_ERROR", [idx[q]], (e01 + e10) / 2)

            c.append("MR", [idx[q]])



    # First round

    syndrome_round(first=True)

    # Detector on each Z-ancilla (compares to deterministic 0 first round)

    for i, q in enumerate(Z_ANC):

        # most recent measurement of this ancilla

        pass  # detectors added below via rec offsets



    # Simpler + robust: use Stim's generated circuit as ground truth scaffold

    # for detector/observable structure, but inject our noise. Rebuild instead:

    return c  # placeholder, replaced below



# The manual detector bookkeeping is error-prone. Use Stim's generated

# circuit as the noiseless scaffold and inject per-component noise instead.



def build_circuit_scaffold(rounds, m):

    """

    Replicate Stim's built-in surface code noise model exactly, but use

    per-qubit PRX/readout errors and per-pair CZ errors instead of a

    single uniform p. Matches Stim conventions:

      - DEPOLARIZE1 after each 1-qubit gate (per-qubit PRX error)

      - DEPOLARIZE2 after each 2-qubit gate (per-pair CZ error)

      - X_ERROR before each measurement (per-qubit readout error)

      - X_ERROR after each reset (per-qubit reset error)

    """

    base = stim.Circuit.generated(

        "surface_code:rotated_memory_z",

        distance=3,

        rounds=rounds,

        after_clifford_depolarization=0.0,

        before_measure_flip_probability=0.0,

        after_reset_flip_probability=0.0,

    ).flattened()



    coords = base.get_final_qubit_coordinates()

    coord_to_qb = {tuple(xy): q for q, xy in {**DATA, **X_ANC, **Z_ANC}.items()}

    def qb_of(idx):

        c = coords.get(idx)

        return coord_to_qb.get(tuple(int(round(v)) for v in c)) if c else None



    avg_prx = float(np.mean([prx_err(m, q) for q in DATA]))

    avg_cz  = 0.005

    avg_ro  = 0.01



    noisy = stim.Circuit()

    for inst in base:

        name = inst.name

        # pass through annotations untouched

        if name in ("QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE",

                    "SHIFT_COORDS", "TICK"):

            noisy.append(inst)

            continue



        targets = inst.targets_copy()



        # X_ERROR BEFORE measurements (before_measure_flip_probability)

        if name in ("M", "MR", "MX", "MZ"):

            for t in targets:

                qb = qb_of(t.value)

                if qb:

                    e01, e10 = readout_err(m, qb)

                    p = (e01 + e10) / 2.0

                else:

                    p = avg_ro

                noisy.append("X_ERROR", [t.value], p)

            noisy.append(inst)

            # X_ERROR AFTER reset part of MR (after_reset_flip_probability)

            if name in ("MR",):

                for t in targets:

                    qb = qb_of(t.value)

                    p = prx_err(m, qb) if qb else avg_prx

                    noisy.append("X_ERROR", [t.value], p)

            continue



        # pure reset

        if name in ("R", "RX", "RZ"):

            noisy.append(inst)

            for t in targets:

                qb = qb_of(t.value)

                p = prx_err(m, qb) if qb else avg_prx

                noisy.append("X_ERROR", [t.value], p)

            continue



        # gates: append then add depolarizing

        noisy.append(inst)

        if name in ("H", "S", "SQRT_X", "SQRT_Y", "SQRT_X_DAG", "SQRT_Y_DAG"):

            for t in targets:

                qb = qb_of(t.value)

                p = prx_err(m, qb) if qb else avg_prx

                noisy.append("DEPOLARIZE1", [t.value], p)

        elif name in ("CX", "CZ", "CNOT", "XCX", "XCZ"):

            for i in range(0, len(targets), 2):

                q1, q2 = targets[i].value, targets[i+1].value

                qb1, qb2 = qb_of(q1), qb_of(q2)

                p = cz_err(m, qb1, qb2) if (qb1 and qb2) else avg_cz

                noisy.append("DEPOLARIZE2", [q1, q2], p)



    return noisy



if __name__ == "__main__":

    m = load_metrics(METRICS_PATH)

    print("Loaded metrics. Sample error rates for your qubits:")

    for q in list(DATA.keys())[:3]:

        e01, e10 = readout_err(m, q)

        print(f"  {q}: PRX={prx_err(m,q):.5f} RO(0->1)={e01:.4f} RO(1->0)={e10:.4f}")

    print("  CZ QB6-QB7:", f"{cz_err(m,'QB6','QB7'):.5f}")



    ROUNDS_LIST = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]

    SHOTS = 100_000



    print("\nGenerating hardware-calibrated data...")

    with h5py.File(f"{OUTPUT_DIR}/d3_zbasis_emerald.h5", "w") as f:

        for rounds in ROUNDS_LIST:

            circuit = build_circuit_scaffold(rounds, m)

            sampler = circuit.compile_detector_sampler()

            syn, obs = sampler.sample(SHOTS, separate_observables=True)

            grp = f.create_group(f"r{rounds}")

            grp.create_dataset("syndromes", data=syn.astype(np.bool_))

            grp.create_dataset("observables", data=obs.astype(np.bool_))

            print(f"  rounds={rounds} | syndromes {syn.shape}")



    print(f"\nDone. Saved to {OUTPUT_DIR}/d3_zbasis_emerald.h5")

