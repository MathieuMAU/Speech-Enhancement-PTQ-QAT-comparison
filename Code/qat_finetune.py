import torch
import time
import csv
import os
from datasets import load_dataset
from tqdm import tqdm
from fonctions import collate_fn, process, compute_loss, convert_gru_to_custom, calibrate_activations
from modules import DeepFilterNet2, Config, CustomGRU
from quantization import enable_lsq_weights, attach_lsq_activations, attach_lsq_activations_from_checkpoint
from pathlib import Path

n_fft = 512
f_df = 5000
epochs = 30
batch_size = 8
C = 64
N = 5
lambdaspec = 1e3
lambdamr = 5e2
bit_widths = [16, 10, 8, 6, 4, 2]

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)


if torch.cuda.is_available():
    gpu_model = torch.cuda.get_device_name(0)
    device = "cuda"
else:
    gpu_model = "CPU"
    device = "cpu"

print(f"Running on {device}")

ds = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")
split = ds["train"].train_test_split(
    test_size=0.10,
    seed=42
)
train = split["train"]
train_loader = torch.utils.data.DataLoader(
    train,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=True,
)


val = split["test"]
val_loader = torch.utils.data.DataLoader(
    val,
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
    n_fft,
    d=1 / sample_rate,
)
# Remove DC and Nyquist bins
freqs = freqs[1:-1]
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

results_dir = Path("results/qat")
results_dir.mkdir(exist_ok=True)

for bits in bit_widths:
    print(f"Starting QAT with {bits}-bit quantization")
    stats_file = results_dir / f"qat_training_stats_{bits}bit.csv"

    model = DeepFilterNet2(C, N, N_df)
    start_epoch = 0
    best_val_loss = float("inf")
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
                "batches_size",
                "gpu_model"
            ])
            checkpoint = torch.load(
                checkpoint_dir / f"best.pt",
                map_location=device,
            )
            model.load_state_dict(checkpoint["model_state_dict"])

            model.encoder.gru = convert_gru_to_custom(model.encoder.gru)
            model.erb_decoder.gru = convert_gru_to_custom(
                model.erb_decoder.gru)
            model.comp_decoder.gru = convert_gru_to_custom(
                model.comp_decoder.gru)

            model = model.to("cpu")
            window = window.to("cpu")
            observers = calibrate_activations(model, train, window, config, 30)
            enable_lsq_weights(model, bits, custom=CustomGRU)
            handles = attach_lsq_activations(
                model, observers, bits, custom=CustomGRU)

    else:
        with open(stats_file, "r") as f:
            last_line = f.readlines()[-1]
            last_epoch = int(last_line.split(",")[0])
            if last_epoch >= epochs:
                print(
                    f"QAT training for {bits}-bit already completed. Skipping.")
                continue
            else:
                print(
                    f"Resuming QAT training for {bits}-bit from epoch {last_epoch + 1}.")
                start_epoch = last_epoch
                best_val_loss = float(last_line.split(",")[4])
                checkpoint = torch.load(
                    checkpoint_dir / f"last_qat_{bits}bit.pt",
                    map_location=device,
                )
                state_dict = checkpoint["model_state_dict"]
                model.encoder.gru = convert_gru_to_custom(model.encoder.gru)
                model.erb_decoder.gru = convert_gru_to_custom(
                    model.erb_decoder.gru)
                model.comp_decoder.gru = convert_gru_to_custom(
                    model.comp_decoder.gru)

                # Recreate LSQ weight parametrizations
                enable_lsq_weights(
                    model,
                    bits,
                    custom=CustomGRU
                )

                # Recreate LSQ activation quantizers and hooks
                handles = attach_lsq_activations_from_checkpoint(
                    model,
                    state_dict,
                    bits,
                    custom=CustomGRU
                )

                # Restore learned FP32 weights and scales
                model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    window = window.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    epoch_times = []
    epoch_peak_memory = []

    for e in tqdm(range(start_epoch, epochs), desc="Epochs", position=0):
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

        for batch in tqdm(train_loader, desc="Training", position=1):
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
            break

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
            for batch in tqdm(val_loader, desc="Validation", position=1):
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
                break

        val_total /= len(val_loader)
        val_mr /= len(val_loader)
        val_spec /= len(val_loader)

        is_best = val_total < best_val_loss

        if is_best:
            best_val_loss = val_total

        checkpoint = {
            "epoch": e + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_total,
            "best_val_loss": best_val_loss
        }

        # Save only when validation improves
        if is_best:
            torch.save(
                checkpoint,
                checkpoint_dir / f"best_qat_{bits}bit.pt"
            )

        # Always save latest checkpoint
        torch.save(
            checkpoint,
            checkpoint_dir / f"last_qat_{bits}bit.pt"
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
                batch_size,
                gpu_model,
            ])
