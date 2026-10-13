import torch


def fake_quantize_symmetric(x, bits):
    qmax = 2 ** (bits - 1) - 1

    max_val = x.abs().max()

    if max_val == 0:
        return x

    scale = max_val / qmax

    x_q = torch.round(x / scale)
    x_q = torch.clamp(x_q, -qmax, qmax)

    # Dequantize
    x_dq = x_q * scale

    return x_dq


def fake_quantize_activation(x, scale, bits=8):

    qmax = 2 ** (bits - 1) - 1

    q = torch.round(x / scale)

    q = torch.clamp(
        q,
        -qmax,
        qmax
    )

    return q * scale
