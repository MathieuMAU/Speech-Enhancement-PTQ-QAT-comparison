import torch
import time
import csv
import os
from datasets import load_dataset
from tqdm.auto import tqdm
from fonctions import collate_fn, process, compute_loss, convert_gru_to_custom, evaluate_model
from modules import DeepFilterNet2, Config, CustomGRU
from pathlib import Path
from quantization import enable_lsq_weights, attach_lsq_activations_from_checkpoint


n_fft = 512
f_df = 5000
epochs = 30
batch_size = 4
C = 64
N = 5
lambdaspec = 1e3
lambdamr = 5e2

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)
stats_file = checkpoint_dir / "training_stats_qat.csv"

if torch.cuda.is_available():
    gpu_model = torch.cuda.get_device_name(0)
    device = "cuda"
else:
    gpu_model = "CPU"
    device = "cpu"

print(f"Running on {gpu_model} ({device})")

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
test = ds["test"]

row = train[0]
samples = row['clean'].get_all_samples()
waveform = samples.data
sample_rate = samples.sample_rate

win_length = int(sample_rate / 1000 * 20)
window = torch.hann_window(win_length)
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
    N_df=N_df,
    higher_precision_modules=[],
    higher_bits=8
)

bit_widths = [16, 10, 8, 6, 4, 2]
checkpoint_dir = Path("checkpoints")
process_dir0 = Path("qat_samples")
process_dir0.mkdir(parents=True, exist_ok=True)
results_dir = Path("results/qat")
results_dir.mkdir(parents=True, exist_ok=True)

for bits in bit_widths:
    print(f"Starting QAT evaluation with {bits}-bit quantization")

    if bits == 4:
        config.higher_precision_modules = [
            "encoder.gru",
            "erb_decoder.gru",
            "comp_decoder.gru"
        ]
        config.higher_bits = 8

    model = DeepFilterNet2(C, N, N_df)

    checkpoint = torch.load(
        checkpoint_dir / f"best_qat_{bits}bit.pt",
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
        custom=CustomGRU,
        config=config
    )

    # Recreate LSQ activation quantizers and hooks
    handles = attach_lsq_activations_from_checkpoint(
        model,
        state_dict,
        bits,
        custom=CustomGRU,
        config=config
    )

    # Restore learned FP32 weights and scales
    model.load_state_dict(state_dict, strict=True)
    process_dir = process_dir0 / f"{bits}bits"
    process_dir.mkdir(parents=True, exist_ok=True)
    # Validation
    sample_df = evaluate_model(
        model_q=model,
        test=test,
        window=window,
        config=config,
        description=f"W{bits}A{bits}",
        save_process=True,
        process_dir=process_dir
    )
    csv_path = results_dir / f"qat_{bits}bits.csv"
    results_dir.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(
        csv_path,
        index=False
    )

    print(f"Results saved to: {csv_path}")

    del model
