from modules import DeepFilterNet2, Config
from fonctions import quantize_model_weights, calibrate_activations, attach_activation_quantizers, evaluate_model, convert_gru_to_custom
import torch
import pandas as pd
from datasets import load_dataset
from pathlib import Path
import time

bit_widths = [16, 10, 8, 6, 4, 2]

device = torch.device("cpu")

ds = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")
split = ds["train"].train_test_split(
    test_size=0.10,
    seed=42
)
train = split["train"]
val = split["test"]
test = ds["test"]

sample_rate = 16000
n_fft = 512
f_df = 5000
C = 64
N = 5
lambdaspec = 1e3
lambdamr = 5e2

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

model = DeepFilterNet2(C, N, N_df).to(device)
model.encoder.gru = convert_gru_to_custom(model.encoder.gru)
model.erb_decoder.gru = convert_gru_to_custom(model.erb_decoder.gru)
model.comp_decoder.gru = convert_gru_to_custom(model.comp_decoder.gru)

checkpoint = torch.load(
    "checkpoints/best.pt",
    map_location=device,
)

model.load_state_dict(checkpoint["model_state_dict"])

model_fp32 = model.cpu()
model_fp32.eval()

window_cpu = window.cpu()

results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)

process_dir0 = Path("ptq")
process_dir0.mkdir(parents=True, exist_ok=True)

for bits in bit_widths:

    print("\n" + "=" * 60)
    print(f"Running W{bits}A{bits}")
    print("=" * 60)

    csv_path = results_dir / f"W{bits}A{bits}.csv"

    if csv_path.exists():
        print(f"W{bits}A{bits} already completed. Skipping.")
        continue

    # Fresh model from FP32
    model_q = quantize_model_weights(
        model_fp32,
        bits=bits
    )
    print(f"Model weights quantized to {bits}-bit.")

    # Calibration: training data only
    start_calibration = time.perf_counter()
    observers = calibrate_activations(
        model_q=model_q,
        calibration_data=train,
        window=window_cpu,
        config=config,
        num_calibration=200
    )

    calibration_duration = time.perf_counter() - start_calibration

    print(f"Calibration W{bits}A{bits}: {calibration_duration:.2f} seconds")
    calibration_csv = results_dir / "calibration_times.csv"

    calibration_df = pd.DataFrame([{
        "bits": bits,
        "configuration": f"W{bits}A{bits}",
        "num_calibration": 200,
        "calibration_seconds": calibration_duration,
        "calibration_minutes": calibration_duration / 60,
    }])

    calibration_df.to_csv(
        calibration_csv,
        mode="a",
        header=not calibration_csv.exists(),
        index=False
    )

    # Enable activation fake quantization
    activation_handles = attach_activation_quantizers(
        model_q=model_q,
        observers=observers,
        bits=bits
    )
    print(f"Activation quantizers attached for W{bits}A{bits}.")

    process_dir = process_dir0 / f"W{bits}A{bits}"
    process_dir.mkdir(parents=True, exist_ok=True)
    # Validation
    sample_df = evaluate_model(
        model_q=model_q,
        test=test,
        window=window_cpu,
        config=config,
        description=f"W{bits}A{bits}",
        save_process=True,
        process_dir=process_dir
    )
    # --------------------------------------------
    # SAVE AFTER EVERY EXPERIMENT
    # --------------------------------------------

    sample_df.to_csv(
        csv_path,
        index=False
    )

    print(f"Results saved to: {csv_path}")

    # Remove activation hooks
    for handle in activation_handles:
        handle.remove()

    del model_q


print("\nAll experiments completed.")
