
# IQM Emerald Data Collection for AQ1 Finetuning



## Instructions for Claude Code



This file instructs how to collect real IQM Emerald syndrome data for

finetuning the AQ1 decoder. Use the Quantum-ECC repo pipeline.



## Prerequisites

- Quantum-ECC repo cloned and set up

- RESONANCE_KEY set in Quantum-ECC/.env

- Python 3.11+ with iqm-client, qiskit, pymatching, h5py installed



## What to collect

- Circuit: d=3 rotated surface code, Z-basis memory

- Placement: optimized surface-17 patch (zero SWAPs verified)

  - Data qubits (9): QB6, QB13, QB22, QB11, QB20, QB29, QB18, QB27, QB36

  - X-ancillas (4): QB7, QB21, QB19, QB35

  - Z-ancillas (4): QB12, QB30, QB10, QB28

  - Always exclude: QB25, QB9 (near-dead qubits)

- Round counts: 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 20

- Shots: 20,000 per round-count

- Backend: emerald (real hardware)



## Output format

HDF5 file: data/d3_zbasis_emerald_real.h5

Structure per group /r{N}/:

  syndromes    shape (20000, N*8)  dtype bool  (Stim detection events)

  observables  shape (20000, 1)    dtype bool  (logical Z flip)



## Safety sequence (always follow this order)

1. Run --sim first (free) to confirm pipeline works

2. Run R=1, 1000 shots (~1.5 credits) to confirm real hardware works

3. Run full sweep with --confirm (~300-500 credits total)



## Script

Use experiments/collect_finetune_data.py from the AQ1 repo.

Copy it into Quantum-ECC/experiments/ and run:



    # Test sim (free):

    python3 experiments/collect_finetune_data.py --sim



    # Real hardware:

    python3 experiments/collect_finetune_data.py --confirm



## After collection

- Save output to data/d3_zbasis_emerald_real.h5

- Report credits used per round-count

- Report any HOT detectors (firing rate > 0.2)

- Share the HDF5 file



## Credit budget

Available: 5000 credits at 0.75 credits/second

R=1 costs ~8.5 credits for 20k shots

Full sweep estimated: 300-500 credits

