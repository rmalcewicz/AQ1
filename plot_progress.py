import sys
import re
import os
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_log(log_path):
    epochs, train_loss, val_loss, train_acc, val_acc, val_ler, lr = [], [], [], [], [], [], []
    with open(log_path) as f:
        for line in f:
            m = re.search(
                r'Epoch\s+(\d+)/\d+.*train loss ([\d.]+) acc ([\d.]+).*val loss ([\d.]+) acc ([\d.]+).*lr ([\d.e+-]+)',
                line)
            if m:
                epochs.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                train_acc.append(float(m.group(3)))
                val_loss.append(float(m.group(4)))
                val_acc.append(float(m.group(5)))
                val_ler.append(1.0 - float(m.group(5)))
                lr.append(float(m.group(6)))
    return epochs, train_loss, val_loss, train_acc, val_acc, val_ler, lr

def plot(log_path, out_path):
    epochs, train_loss, val_loss, train_acc, val_acc, val_ler, lr = parse_log(log_path)
    if not epochs:
        print("No epoch data found yet")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'AQ1 Training Progress — {os.path.basename(log_path)}', fontsize=13)

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, 'b-o', ms=4, label='Train loss')
    ax.plot(epochs, val_loss, 'r-o', ms=4, label='Val loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('BCE Loss')
    ax.set_title('Loss'); ax.legend(); ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, train_acc, 'b-o', ms=4, label='Train acc')
    ax.plot(epochs, val_acc, 'r-o', ms=4, label='Val acc')
    ax.axhline(y=max(val_acc[0], 0.827), color='gray', linestyle='--',
               alpha=0.5, label='Majority baseline ~82.7%')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy'); ax.legend(); ax.grid(True, alpha=0.3)

    # Logical error rate
    ax = axes[1, 0]
    ax.plot(epochs, val_ler, 'g-o', ms=4, label='AQ1 val LER')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Logical Error Rate')
    ax.set_title('Logical Error Rate (lower = better)')
    ax.legend(); ax.grid(True, alpha=0.3)

    # LR
    ax = axes[1, 1]
    ax.plot(epochs, lr, 'm-o', ms=4)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"Saved plot to {out_path}")
    print(f"  Epochs: {epochs[0]}-{epochs[-1]}")
    print(f"  Best val LER: {min(val_ler):.4f} at epoch {epochs[val_ler.index(min(val_ler))]}")
    print(f"  Latest LR: {lr[-1]:.2e}")

if __name__ == '__main__':
    # Find most recent log or use argument
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        log_dir = os.environ.get("LOG_DIR", "logs")
        logs = glob.glob(f'{log_dir}/train_*.log')
        logs += glob.glob(f'{log_dir}/test_*.log')
        if not logs:
            print("No log files found")
            sys.exit(1)
        log_path = max(logs, key=os.path.getmtime)
        print(f"Using most recent log: {log_path}")

    out_path = log_path.replace('.log', '_progress.png')
    plot(log_path, out_path)
