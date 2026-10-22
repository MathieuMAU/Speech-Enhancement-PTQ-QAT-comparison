import torch
from torch.nn.utils import parametrize


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


class LSQFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale, qmin, qmax, grad_factor):

        x_scaled = x / scale
        x_clamped = x_scaled.clamp(qmin, qmax)
        x_rounded = x_clamped.round()

        ctx.save_for_backward(x_scaled, x_rounded)
        ctx.qmin = qmin
        ctx.qmax = qmax
        ctx.grad_factor = grad_factor

        return x_rounded * scale

    @staticmethod
    def backward(ctx, grad_output):

        x_scaled, x_rounded = ctx.saved_tensors
        qmin = ctx.qmin
        qmax = ctx.qmax

        # STE gradient for input
        inside = (x_scaled >= qmin) & (x_scaled <= qmax)
        grad_x = grad_output * inside

        # LSQ gradient for scale
        grad_s = torch.where(
            x_scaled < qmin,
            torch.full_like(x_scaled, qmin),
            torch.where(
                x_scaled > qmax,
                torch.full_like(x_scaled, qmax),
                x_rounded - x_scaled
            )
        )

        grad_s = (
            grad_s * grad_output
        ).sum() * ctx.grad_factor

        return grad_x, grad_s, None, None, None


class LSQQuantizer(torch.nn.Module):

    def __init__(self, bits, init_scale=1.0):
        super().__init__()

        self.bits = bits

        self.qmax = 2 ** (bits - 1) - 1
        self.qmin = -self.qmax

        self.scale = torch.nn.Parameter(
            torch.tensor(float(init_scale))
        )

    def forward(self, x):

        # Ensure scale remains positive
        scale = self.scale.abs().clamp(min=1e-8)

        grad_factor = 1.0 / (
            (x.numel() * self.qmax) ** 0.5
        )

        return LSQFunction.apply(
            x,
            scale,
            self.qmin,
            self.qmax,
            grad_factor
        )


def enable_lsq_weights(model, bits, custom):

    for module in list(model.modules()):

        if isinstance(
            module,
            (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)
        ):
            weight = module.weight.detach()

            init_scale = (
                weight.abs().max().item()
                / (2 ** (bits - 1) - 1)
            )

            parametrize.register_parametrization(
                module,
                "weight",
                LSQQuantizer(bits, init_scale)
            )

        elif isinstance(module, custom):

            for layer in range(module.num_layers):

                for name in (
                    f"weight_ih_l{layer}",
                    f"weight_hh_l{layer}"
                ):
                    weight = getattr(module, name).detach()

                    init_scale = (
                        weight.abs().max().item()
                        / (2 ** (bits - 1) - 1)
                    )

                    parametrize.register_parametrization(
                        module,
                        name,
                        LSQQuantizer(bits, init_scale)
                    )


def attach_lsq_activations(model, observers, bits, custom):

    handles = []

    for name, module in list(model.named_modules()):

        if isinstance(module, custom):

            module.lsq_inputs = torch.nn.ModuleList()
            module.lsq_hidden = torch.nn.ModuleList()

            for layer in range(module.num_layers):

                input_scale = observers[
                    f"{name}.layer{layer}.input"
                ].get_scale(bits=bits)

                hidden_scale = observers[
                    f"{name}.layer{layer}.hidden"
                ].get_scale(bits=bits)

                module.lsq_inputs.append(
                    LSQQuantizer(bits, input_scale)
                )

                module.lsq_hidden.append(
                    LSQQuantizer(bits, hidden_scale)
                )

            continue

        if (
            name not in observers
            or not isinstance(
                module,
                (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)
            )
        ):
            continue

        init_scale = observers[name].get_scale(bits=bits)

        module.lsq_input = LSQQuantizer(
            bits, init_scale
        )

        def hook(module, inputs):
            x_q = module.lsq_input(inputs[0])
            return (x_q,) + inputs[1:]

        handles.append(
            module.register_forward_pre_hook(hook)
        )

    return handles


def attach_lsq_activations_from_checkpoint(
    model, state_dict, bits, custom
):
    handles = []

    for name, module in list(model.named_modules()):

        if isinstance(module, custom):

            module.lsq_inputs = torch.nn.ModuleList()
            module.lsq_hidden = torch.nn.ModuleList()

            for layer in range(module.num_layers):

                input_key = f"{name}.lsq_inputs.{layer}.scale"
                hidden_key = f"{name}.lsq_hidden.{layer}.scale"

                input_scale = state_dict[input_key].item()
                hidden_scale = state_dict[hidden_key].item()

                module.lsq_inputs.append(
                    LSQQuantizer(bits, input_scale)
                )
                module.lsq_hidden.append(
                    LSQQuantizer(bits, hidden_scale)
                )

            continue

        if not isinstance(
            module,
            (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)
        ):
            continue

        scale_key = f"{name}.lsq_input.scale"

        if scale_key not in state_dict:
            continue

        init_scale = state_dict[scale_key].item()

        module.lsq_input = LSQQuantizer(
            bits, init_scale
        )

        def hook(module, inputs):
            x_q = module.lsq_input(inputs[0])
            return (x_q,) + inputs[1:]

        handles.append(
            module.register_forward_pre_hook(hook)
        )

    return handles
