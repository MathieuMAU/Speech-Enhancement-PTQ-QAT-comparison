from torch import stft as torch_stft
import torch
import numpy as np
import torch.nn.functional as F
from modules import Config


def collate_fn(batch):
    clean = []
    noisy = []

    for row in batch:
        clean_waveform = row["clean"].get_all_samples().data
        noisy_waveform = row["noisy"].get_all_samples().data

        # Usually [channels, samples]
        clean_waveform = clean_waveform.squeeze(0).float()
        noisy_waveform = noisy_waveform.squeeze(0).float()

        clean.append(clean_waveform)
        noisy.append(noisy_waveform)

    # Handle different utterance lengths
    clean = torch.nn.utils.rnn.pad_sequence(
        clean,
        batch_first=True
    )

    noisy = torch.nn.utils.rnn.pad_sequence(
        noisy,
        batch_first=True
    )

    return {
        "clean": clean,
        "noisy": noisy,
    }


def compute_stft(waveform, win_length, window, hop_length, n_fft=512):
    ft = torch_stft(waveform, n_fft=n_fft, win_length=win_length,
                    hop_length=hop_length, window=window, return_complex=True)
    ft = ft[:, 1:-1, :]
    return ft


def normalize_power_spectrum(ft, sample_rate, hop_length, decay_time=1, eps=1e-10):
    power = torch.abs(ft) ** 2
    power_db = 10 * torch.log10(power + 1e-10)
    Xnorm = torch.empty_like(
        power_db, device=power_db.device, dtype=power_db.dtype)

    mean = torch.zeros(
        power_db.shape[0], power_db.shape[1], device=power_db.device, dtype=power_db.dtype)
    second_moment = torch.zeros(
        power_db.shape[0], power_db.shape[1], device=power_db.device, dtype=power_db.dtype)

    dt = hop_length / sample_rate
    alpha = np.exp(-dt/decay_time)

    for t in range(0, power_db.shape[2]):
        x = power_db[:, :, t]

        mean = alpha * mean + (1 - alpha) * x

        second_moment = (alpha * second_moment + (1 - alpha) * x.square())
        variance = second_moment - mean.square()

        variance = variance.clamp_min(eps)

        Xnorm[:, :, t] = (x - mean) / torch.sqrt(variance)
    return Xnorm


def normalize_complex_features(ft, sample_rate, df_indices, hop_length, f_df=5000, decay_time=1, eps=1e-10):
    B, F, T = ft.shape

    ft = ft[:, df_indices, :]

    dt = hop_length / sample_rate
    alpha = torch.exp(
        torch.tensor(
            -dt / decay_time,
            device=ft.device,
            dtype=torch.float32,
        )
    ).to(ft.real.dtype)

    # Running second moment of complex magnitude
    second_moment = torch.zeros(
        B,
        ft.shape[1],
        device=ft.device,
        dtype=ft.real.dtype,
    )

    XDF = torch.empty_like(ft)

    for t in range(T):
        x = ft[:, :, t]

        power = x.abs().square()

        second_moment = (
            alpha * second_moment
            + (1 - alpha) * power
        )

        XDF[:, :, t] = x / torch.sqrt(
            second_moment + eps
        )

    return XDF


def process(waveform, window, config: Config):
    ft = compute_stft(waveform, config.win_length, window,
                      config.hop_length, config.n_fft)
    Xnorm = normalize_power_spectrum(ft, config.sample_rate, config.hop_length)

    erb_idx = get_erb_fb(32, config.sample_rate, config.n_fft)
    erb_idx = torch.tensor(erb_idx).view(
        1, -1, 1).expand(Xnorm.shape[0], -1, Xnorm.shape[2]).to(Xnorm.device)
    log_power_erb = torch.zeros(
        (Xnorm.shape[0], 32, Xnorm.shape[2]),
        dtype=Xnorm.dtype,
        device=Xnorm.device)
    log_power_erb.scatter_add_(1, erb_idx, Xnorm)

    Xdf = normalize_complex_features(
        ft, config.sample_rate, config.df_indices, config.hop_length)
    Xdf = torch.stack([
        Xdf.real,
        Xdf.imag
    ], dim=1)
    return log_power_erb, Xdf, ft


def get_erb_fb(nb_bands, sample_rate, n_fft):
    def hz_to_erb(f): return 21.4 * np.log10(4.37e-3 * f + 1)
    def erb_to_hz(e): return (10 ** (e / 21.4) - 1) / 4.37e-3
    fmin = sample_rate / n_fft
    fmax = sample_rate / 2 - fmin
    erb = np.linspace(hz_to_erb(fmin), hz_to_erb(fmax), nb_bands)
    edges = erb_to_hz(erb)

    freqs = np.fft.rfftfreq(512, 1 / 16000)[1:-1]
    band_idx = np.digitize(freqs, edges) - 1

    return band_idx


def apply_deep_filter(YG, C, l=2):
    """
    YG: [B, F, T] complex
    C:  [B, N, F, T] complex
    """
    B, F, T = YG.shape
    N = C.shape[1]

    Y = torch.zeros_like(YG)

    for i in range(N):
        shift = i - l

        if shift >= 0:
            Y[:, :, shift:] += (
                C[:, i, :, shift:] *
                YG[:, :, :T-shift]
            )
        else:
            d = -shift
            Y[:, :, :T-d] += (
                C[:, i, :, :T-d] *
                YG[:, :, d:]
            )

    return Y


def compressed_complex_stft(X, c=0.3, eps=1e-8):
    """
    X: complex STFT [B, F, T]
    """
    mag = torch.abs(X)

    # magnitude compression
    mag_c = mag.clamp_min(eps).pow(c)

    # preserve phase
    phase = X / mag.clamp_min(eps)

    return mag_c * phase


def mr_spectrogram_loss(
    y,
    s,
    sample_rate,
    windows_ms=(5, 10, 20, 40),
    c=0.3,
):
    """
    y: predicted/enhanced waveform [B, T]
    s: clean/reference waveform [B, T]
    """

    loss = 0.0

    for window_ms in windows_ms:

        win_length = round(sample_rate * window_ms / 1000)

        hop_length = win_length // 4

        Y = torch.stft(
            y,
            n_fft=win_length,
            win_length=win_length,
            hop_length=hop_length,
            window=torch.hann_window(win_length, device=y.device),
            return_complex=True
        )

        S = torch.stft(
            s,
            n_fft=win_length,
            win_length=win_length,
            hop_length=hop_length,
            window=torch.hann_window(win_length, device=y.device),
            return_complex=True,
        )

        # compressed magnitudes
        Y_mag_c = torch.abs(Y).clamp_min(1e-8).pow(c)
        S_mag_c = torch.abs(S).clamp_min(1e-8).pow(c)

        # magnitude loss
        L_mag = torch.linalg.vector_norm(
            Y_mag_c - S_mag_c
        )

        # compressed complex loss
        Y_c = compressed_complex_stft(Y, c=c)
        S_c = compressed_complex_stft(S, c=c)

        L_complex = torch.linalg.vector_norm(
            Y_c - S_c
        )

        loss = loss + L_mag * L_complex
    return loss


def lspec(Y, S, c=0.6):
    """
    Y: predicted complex STFT, [B, F, T]
    S: target complex STFT,    [B, F, T]
    """

    # Magnitudes
    Y_mag = torch.abs(Y).clamp_min(1e-8)
    S_mag = torch.abs(S).clamp_min(1e-8)

    # Compressed magnitudes
    Y_mag_c = Y_mag.pow(c)
    S_mag_c = S_mag.pow(c)

    # Magnitude loss
    L_mag = torch.linalg.vector_norm(Y_mag_c - S_mag_c)

    # Phase-aware compressed complex spectra
    Y_phase = Y / torch.clamp(Y_mag, min=1e-12)
    S_phase = S / torch.clamp(S_mag, min=1e-12)

    Y_comp = Y_mag_c * Y_phase
    S_comp = S_mag_c * S_phase

    # Complex / phase-aware loss
    L_phase = torch.linalg.vector_norm(Y_comp - S_comp)

    return L_mag + L_phase


def compute_loss(G_erb, C_df, clean, ft, window, config: Config):
    G_prime = G_erb * torch.sin(torch.pi / 2 * G_erb)
    beta = 0.02
    G_pf = (1 + beta) * G_erb / (
        1 + beta + G_prime
    )
    G = F.interpolate(
        G_pf,
        size=(255, G_erb.shape[-1]),
        mode="bilinear",
        align_corners=False,
    )
    G = G.squeeze(1)
    YG = G * ft
    YG_df = YG[:, config.df_indices, :]

    C_real = C_df[:, :config.N]
    C_imag = C_df[:, config.N:]
    C_df_comp = torch.complex(C_real, C_imag)

    Y_df = apply_deep_filter(YG_df, C_df_comp, l=2)
    Y_final = YG.clone()
    Y_final[:, :config.N_df, :] = Y_df

    B, _, T = Y_final.shape

    Y_full = torch.zeros(
        B, config.n_fft//2 + 1, T,
        dtype=Y_final.dtype,
        device=Y_final.device
    )
    Y_full[:, 0, :] = 0
    Y_full[:, 1:-1, :] = Y_final
    Y_full[:, -1, :] = 0
    y = torch.istft(
        Y_full,
        n_fft=512,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=window,
        length=clean.shape[-1],
    )
    loss_mr = mr_spectrogram_loss(y, clean, config.sample_rate)
    S = torch.stft(clean, n_fft=config.n_fft,
                   win_length=config.win_length, hop_length=config.hop_length,
                   window=torch.hann_window(
                       config.win_length, device=clean.device),
                   return_complex=True)
    loss_spec = lspec(Y_full, S)
    return loss_mr, loss_spec
