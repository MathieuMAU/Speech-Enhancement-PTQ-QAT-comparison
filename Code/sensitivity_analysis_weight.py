from modules import DeepFilterNet2, Config, CustomGRU
from fonctions import quantize_model_weights, calibrate_activations, attach_activation_quantizers, evaluate_model, convert_gru_to_custom
import torch
import pandas as pd
from datasets import load_dataset
from pathlib import Path
import time


higher_precision_bits = 16
bit_widths = [10, 8, 6, 4, 2]

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
    N_df=N_df,
    higher_precision_modules=[],
    higher_bits=8
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

results_dir = Path("results/sensitivity_analysis")
results_dir.mkdir(parents=True, exist_ok=True)

modules_name = [name for name, module in model.named_modules() if isinstance(
    module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear, CustomGRU))]

output_path = results_dir / "sensitivity_analysis_results.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)

# Load completed evaluations if resuming
if output_path.exists():
    results = pd.read_csv(output_path)
else:
    results = pd.DataFrame(
        columns=["id", "module", "bits", "pesq", "stoi", "evaluation_time"]
    )


for module_name in modules_name:
    for bit in bit_widths:
        # Skip evaluations already completed
        completed = results[
            (results["module"] == module_name)
            & (results["bits"] == bit)
        ]

        if not completed.empty:
            print(f"Skipping {module_name} at {bit} bits")
            continue

        print(f"Quantizing {module_name} to {bit} bits...")

        time_start = time.time()
        model_quantized = quantize_model_weights(
            model_fp32,
            bits=bit,
            lower_precision_modules=[module_name],
            lower_bits=bit,
            higher_bits=higher_precision_bits
        )

        model_quantized.eval()

        sample_df = evaluate_model(
            model_q=model_quantized,
            test=val,
            window=window_cpu,
            config=config,
            description=f"Evaluating {module_name} - {bit} bits",
            save_process=False,
            process_dir=None
        )

        time_end = time.time()
        sample_df = sample_df[["id", "pesq", "stoi"]].copy()
        sample_df["module"] = module_name
        sample_df["bits"] = bit
        sample_df["evaluation_time"] = time_end - time_start
        results = pd.concat(
            [results, sample_df],
            ignore_index=True
        )

        results.to_csv(output_path, index=False)

        print(f"Saved {module_name} at {bit} bits")

        del model_quantized
