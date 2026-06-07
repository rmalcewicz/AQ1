
# AQ1 — AlphaQubit Decoder for IQM Emerald

Reimplementation of AQ1 (AlphaQubit) trained on d=3 rotated surface code
syndrome data and finetuned on real IQM Emerald hardware.

## Status

Training complete on LUMI supercomputer (AMD MI250X GPUs).
See the accompanying paper for results.

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

## Usage

**Generate training data (Tier 1 — uniform noise):**
```bash
python3 generate_data_full.py         # 60M samples, 20 round counts x 6 noise levels
```

**Generate hardware-calibrated data (Tier 2):**
```bash
# Requires emerald_metrics.json fetched via fetch_iqm_calibration.py
METRICS_PATH=data/emerald_metrics.json python3 generate_data_iqm_full.py
```

**Train:**
```bash
python3 train.py --data data/d3_zbasis_full.h5 --lr 3e-4 --epochs 50
```

**Finetune on your own hardware data:**
```bash
python3 finetune_example.py \
    --pretrained checkpoints/best_model.pt \
    --data data/your_hardware_data.h5 \
    --output checkpoints/finetuned.pt
```

**Plot training progress:**
```bash
python3 plot_progress.py logs/train_log.txt
```

## Dependencies

```
stim
torch
h5py
numpy
matplotlib
iqm-client  # for fetch_iqm_calibration.py only
```

## SLURM jobs (LUMI)

Job scripts are in `jobs/`. Update `--account` to your project ID before submitting.
