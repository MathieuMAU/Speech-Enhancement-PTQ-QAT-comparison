import torch
import time
import csv
import os
from datasets import load_dataset
from tqdm import tqdm
from fonctions import collate_fn, process, compute_loss
from modules import DeepFilterNet2, Config
from pathlib import Path


n_fft = 512
f_df = 5000
epochs = 20
batch_size = 8
C = 64
N = 5
lambdaspec = 1e3
lambdamr = 5e2

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)
stats_file = checkpoint_dir / "training_stats.csv"

# Create the file and header only if it doesn't already exist
if not os.path.exists(stats_file):
    with open(stats_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_mr_loss",
            "train_spec_loss",
            "val_loss",
            "val_mr_loss",
            "val_spec_loss",
            "epoch_time_s",
            "peak_train_gpu_memory_mb",
            "learning_rate",
            "num_batches"
        ])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

ds = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")
train = ds['train']
train_loader = torch.utils.data.DataLoader(
    train,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=True,
)


test = ds['test']
test_loader = torch.utils.data.DataLoader(
    test,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=True,
)

row = train[0]
samples = row['clean'].get_all_samples()
waveform = samples.data
sample_rate = samples.sample_rate

win_length = int(sample_rate / 1000 * 20)
window = torch.hann_window(win_length).to(device)
hop_length = win_length // 2
freqs = torch.fft.rfftfreq(
    n_fft-3,
    d=1 / sample_rate,
)
df_indices = freqs <= f_df
N_df = df_indices.sum().item()

config = Config(
    sample_rate=sample_rate,
    n_fft=n_fft,
    f_df=f_df,
    N=N,
    lambdamr=lambdamr,
    lambdaspec=lambdaspec,
    hop_length=hop_length,
    win_length=win_length,
    df_indices=df_indices,
    N_df=N_df
)

model = DeepFilterNet2(C, N, N_df).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.9
)

epoch_times = []
epoch_peak_memory = []

best_val_loss = float("inf")
for e in tqdm(range(epochs)):
    learning_rate = optimizer.param_groups[0]["lr"]

    model.train()
    train_total = 0.0
    train_mr = 0.0
    train_spec = 0.0

    # Reset peak memory statistics for this epoch
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    for batch in tqdm(train_loader, desc="Training"):
        clean = batch["clean"].to(device, non_blocking=True)
        noisy = batch["noisy"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        Xnorm, Xdf, ft = process(noisy, window, config)
        G_erb, C_df = model(Xnorm, Xdf)
        loss_mr, loss_spec = compute_loss(
            G_erb, C_df, clean, ft, window, config)

        loss = config.lambdamr*loss_mr + config.lambdaspec*loss_spec
        train_total += loss.item()
        train_mr += loss_mr.item()
        train_spec += loss_spec.item()
        loss.backward()

        optimizer.step()

    scheduler.step()

    # Wait for all CUDA operations to finish before stopping timer
    if device == "cuda":
        torch.cuda.synchronize()
        # Peak GPU memory during training for this epoch
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    else:
        peak_memory = 0

    epoch_time = time.perf_counter() - start_time

    train_total = train_total / len(train_loader)
    train_mr = train_mr / len(train_loader)
    train_spec = train_spec / len(train_loader)

    epoch_times.append(epoch_time)
    epoch_peak_memory.append(peak_memory)

    model.eval()

    val_total = 0.0
    val_spec = 0.0
    val_mr = 0.0

    with torch.no_grad():
        for batch in test_loader:
            clean = batch["clean"].to(device, non_blocking=True)
            noisy = batch["noisy"].to(device, non_blocking=True)

            Xnorm, Xdf, ft = process(noisy, window, config)
            G_erb, C_df = model(Xnorm, Xdf)

            loss_mr, loss_spec = compute_loss(
                G_erb, C_df, clean, ft, window, config
            )
            loss = config.lambdamr*loss_mr + config.lambdaspec*loss_spec
            val_total += loss.item()
            val_mr += loss_mr.item()
            val_spec += loss_spec.item()

    val_total /= len(test_loader)
    val_mr /= len(test_loader)
    val_spec /= len(test_loader)

    checkpoint = {
        "epoch": e + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": val_total,
    }

    # Always save latest checkpoint
    torch.save(
        checkpoint,
        checkpoint_dir / "last.pt"
    )

    # Save only when validation improves
    if val_total < best_val_loss:
        best_val_loss = val_total

        torch.save(
            checkpoint,
            checkpoint_dir / "best.pt"
        )

    with open(stats_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            e + 1,
            train_total,
            train_mr,
            train_spec,
            val_total,
            val_mr,
            val_spec,
            epoch_time,
            peak_memory,
            learning_rate,
            batch_size
        ])

    print(
        f"Epoch {e + 1}/{epochs} | "
        f"train={train_total:.4e} | "
        f"val={val_total:.4e} | "
        f"time={epoch_time:.2f}s | "
        f"peak GPU={peak_memory:.2f} MB"
    )
