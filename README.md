
# AQ1 — AlphaQubit Decoder for IQM Emerald



Reimplementation of AQ1 (AlphaQubit) trained on d=3 rotated surface code

syndrome data and finetuned on real IQM Emerald hardware.



## Status

🔄 Training in progress



## Architecture

Recurrent transformer decoder: StabilizerEmbedding → SyndromeTransformer → GRU → Readout

1.1M parameters, d=3 rotated surface code Z-basis memory



## Pipeline

- Tier 1: Uniform Stim pretraining (60M samples)

- Tier 2: Emerald-calibrated Stim (10M samples)  

- Tier 3: Real IQM Emerald finetuning (220k shots)



## Hardware

- Training: LUMI supercomputer (AMD MI250X GPUs)

- QPU: IQM Emerald (54-qubit square lattice)



See COLLECT_DATA.md for data collection instructions.

