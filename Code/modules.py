import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from torch.utils.data import Dataset
from pathlib import Path
import torchaudio
from quantization import fake_quantize_activation


class Conv_block(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1)):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 3), stride=(2, 1), padding=(1, 0), output_padding=(1, 0)):
        super().__init__()

        self.tconv = torch.nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )

        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.tconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class GLinear(torch.nn.Module):
    def __init__(self, in_features=512, out_features=512, groups=8):
        super().__init__()

        assert in_features % groups == 0
        assert out_features % groups == 0

        self.groups = groups
        self.in_group_features = in_features // groups
        self.out_group_features = out_features // groups

        self.linears = torch.nn.ModuleList([
            torch.nn.Linear(self.in_group_features, self.out_group_features)
            for _ in range(groups)
        ])

    def forward(self, x):
        # x: [B, F, T]

        chunks = x.chunk(self.groups, dim=1)

        outputs = [
            linear(chunk.transpose(1, 2)).transpose(1, 2)
            for linear, chunk in zip(self.linears, chunks)
        ]

        return torch.cat(outputs, dim=1)


class Encoder(torch.nn.Module):
    def __init__(self, C, N_df):
        super().__init__()
        self.erb_conv1 = Conv_block(
            in_channels=1, out_channels=C, kernel_size=(3, 3), stride=(1, 1))
        self.erb_conv2 = Conv_block(
            in_channels=C, out_channels=C, kernel_size=(1, 3), stride=(2, 1))
        self.erb_conv3 = Conv_block(
            in_channels=C, out_channels=C, kernel_size=(1, 3), stride=(2, 1))
        self.erb_conv4 = Conv_block(
            in_channels=C, out_channels=C, kernel_size=(1, 3), stride=(2, 1))

        self.comp_conv1 = Conv_block(
            in_channels=2, out_channels=C*2, kernel_size=(3, 3), stride=(1, 1))
        self.comp_conv2 = Conv_block(
            in_channels=C*2, out_channels=C*2, kernel_size=(1, 3), stride=(2, 1))
        self.comp_glinear = GLinear(2*C*N_df//2, C*4, 8)

        self.group_glinear = GLinear(C*4*2, C*4, 8)
        self.gru = torch.nn.GRU(input_size=C*4, hidden_size=C*4,
                                num_layers=1, batch_first=True, bidirectional=False)

    def forward(self, Xnorm, Xdf):
        Xnorm = Xnorm.unsqueeze(1)  # Add channel dimension
        x_erb = F.pad(Xnorm, (1, 1, 1, 1))
        x_erb1 = self.erb_conv1(x_erb)
        x_erb2 = F.pad(x_erb1, (1, 1, 0, 0))
        x_erb2 = self.erb_conv2(x_erb2)
        x_erb3 = F.pad(x_erb2, (1, 1, 0, 0))
        x_erb3 = self.erb_conv3(x_erb3)
        x_erb4 = F.pad(x_erb3, (1, 1, 0, 0))
        x_erb4 = self.erb_conv4(x_erb4)

        x_comp = F.pad(Xdf, (1, 1, 1, 1))
        x_comp1 = self.comp_conv1(x_comp)
        x_comp2 = F.pad(x_comp1, (1, 1, 0, 0))
        x_comp2 = self.comp_conv2(x_comp2)
        x_comp2 = x_comp2.flatten(start_dim=1, end_dim=2)
        x_comp2 = self.comp_glinear(x_comp2)

        x_group = torch.cat(
            [x_erb4.flatten(start_dim=1, end_dim=2), x_comp2], dim=1)
        x_group = self.group_glinear(x_group)
        x_group = self.gru(x_group.transpose(1, 2))[0].transpose(1, 2)

        return x_erb1, x_erb2, x_erb3, x_erb4, x_comp1, x_group


class ERB_decoder(torch.nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        self.gru = torch.nn.GRU(input_size=C*4, hidden_size=C*4,
                                num_layers=2, batch_first=True, bidirectional=False)
        self.glinear = GLinear(C*4, C*4, 8)
        self.tconv1 = TConvBlock(in_channels=2*C, out_channels=C, kernel_size=(
            1, 3), stride=(2, 1), padding=(0, 1), output_padding=(1, 0))
        self.tconv2 = TConvBlock(in_channels=2*C, out_channels=C, kernel_size=(
            1, 3), stride=(2, 1), padding=(0, 1), output_padding=(1, 0))
        self.tconv3 = TConvBlock(in_channels=2*C, out_channels=C, kernel_size=(
            1, 3), stride=(2, 1), padding=(0, 1), output_padding=(1, 0))

        self.pconv1 = torch.nn.Conv2d(C, C, kernel_size=(1, 1))
        self.pconv2 = torch.nn.Conv2d(C, C, kernel_size=(1, 1))
        self.pconv3 = torch.nn.Conv2d(C, C, kernel_size=(1, 1))
        self.pconv4 = torch.nn.Conv2d(C, C, kernel_size=(1, 1))

        self.conv = torch.nn.Conv2d(
            in_channels=2*C, out_channels=1, kernel_size=(3, 3), stride=(1, 1))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x_group, x_erb4, x_erb3, x_erb2, x_erb1):
        x_erb4 = self.pconv1(x_erb4)
        x_erb3 = self.pconv2(x_erb3)
        x_erb2 = self.pconv3(x_erb2)
        x_erb1 = self.pconv4(x_erb1)
        x = self.gru(x_group.transpose(1, 2))[0].transpose(1, 2)
        x = self.glinear(x)
        x = x.reshape(-1, self.C, 4, x.shape[2])
        x = torch.cat([x, x_erb4], dim=1)
        x = self.tconv1(x)
        x = torch.cat([x, x_erb3], dim=1)
        x = self.tconv2(x)
        x = torch.cat([x, x_erb2], dim=1)
        x = self.tconv3(x)
        x = torch.cat([x, x_erb1], dim=1)
        x = F.pad(x, (2, 0, 1, 1))
        x = self.conv(x)
        x = self.sigmoid(x)
        return x


class Comp_decoder(torch.nn.Module):
    def __init__(self, C, N, N_df):
        super().__init__()
        self.C = C
        self.N = N
        self.N_df = N_df
        self.gru = torch.nn.GRU(input_size=C*4, hidden_size=C*N,
                                num_layers=2, batch_first=True, bidirectional=False)
        self.glinear1 = GLinear(C*4, C*4, 8)
        self.glinear2 = GLinear(C*N, 2*N*N_df, N*2)

        self.pconv = torch.nn.Conv2d(2*C, 2*N, kernel_size=(1, 1))

        self.conv = torch.nn.Conv2d(
            in_channels=2*C, out_channels=2*N, kernel_size=(3, 3), stride=(1, 1))

    def forward(self, x_group, x_comp1):
        x_comp1 = self.pconv(x_comp1)
        x = self.glinear1(x_group)
        x = self.gru(x.transpose(1, 2))[0].transpose(1, 2)
        x = self.glinear2(x)
        x = x.reshape(-1, self.N*2, self.N_df, x.shape[2])
        x += x_comp1
        return x


class DeepFilterNet2(torch.nn.Module):
    def __init__(self, C, N, N_df):
        super().__init__()

        self.encoder = Encoder(C, N_df)
        self.erb_decoder = ERB_decoder(C)
        self.comp_decoder = Comp_decoder(C, N, N_df)

    def forward(self, Xnorm, Xdf):

        x_erb1, x_erb2, x_erb3, x_erb4, x_comp1, x_group = self.encoder(
            Xnorm, Xdf)
        G_erb = self.erb_decoder(x_group, x_erb4, x_erb3, x_erb2, x_erb1)
        C_df = self.comp_decoder(x_group, x_comp1)
        return G_erb, C_df


@dataclass
class Config:
    sample_rate: int
    n_fft: int
    f_df: float
    N: int
    lambdaspec: int
    lambdamr: int
    hop_length: int
    win_length: int

    df_indices: torch.Tensor
    N_df: int


class EnvironmentalDataset(Dataset):
    def __init__(self, dataframe, clean_dir, noisy_dir):
        self.df = dataframe.reset_index(drop=True)
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        clean, sr_clean = torchaudio.load(
            self.clean_dir / row["clean_file"]
        )

        noisy, sr_noisy = torchaudio.load(
            self.noisy_dir / row["noisy_file"]
        )

        assert sr_clean == sr_noisy == 16000

        return {
            "clean": clean,
            "noisy": noisy,
            "sample_rate": sr_clean,
            "filename": row["clean_file"]
        }


class ActivationObserver:
    def __init__(self):
        self.max_abs = 0.0

    def update(self, x):
        if not torch.is_tensor(x):
            return

        current_max = x.detach().abs().max().item()

        if current_max > self.max_abs:
            self.max_abs = current_max

    def get_scale(self, bits=8):
        qmax = 2 ** (bits - 1) - 1

        return max(self.max_abs / qmax, 1e-12)


class CustomGRU(torch.nn.Module):
    """
    Custom implementation of torch.nn.GRU for:

        batch_first=True
        bidirectional=False

    Supports multiple layers and reproduces PyTorch's GRU equations.

    Input:
        x:  [B, T, input_size]

    Optional:
        h0: [num_layers, B, hidden_size]

    Returns:
        output: [B, T, hidden_size]
        h_n:    [num_layers, B, hidden_size]
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=True,
    ):
        super().__init__()

        if not batch_first:
            raise NotImplementedError(
                "This implementation currently assumes batch_first=True."
            )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first

        # ----------------------------------------------------
        # Create parameters with the SAME naming convention
        # as torch.nn.GRU.
        #
        # This makes transferring weights straightforward.
        # ----------------------------------------------------

        for layer in range(num_layers):

            layer_input_size = (
                input_size
                if layer == 0
                else hidden_size
            )

            # PyTorch combines the three gates:
            #
            # reset
            # update
            # new
            #
            # => 3 * hidden_size

            weight_ih = torch.nn.Parameter(
                torch.empty(
                    3 * hidden_size,
                    layer_input_size,
                )
            )

            weight_hh = torch.nn.Parameter(
                torch.empty(
                    3 * hidden_size,
                    hidden_size,
                )
            )

            self.register_parameter(
                f"weight_ih_l{layer}",
                weight_ih,
            )

            self.register_parameter(
                f"weight_hh_l{layer}",
                weight_hh,
            )

            if bias:

                bias_ih = torch.nn.Parameter(
                    torch.empty(
                        3 * hidden_size
                    )
                )

                bias_hh = torch.nn.Parameter(
                    torch.empty(
                        3 * hidden_size
                    )
                )

                self.register_parameter(
                    f"bias_ih_l{layer}",
                    bias_ih,
                )

                self.register_parameter(
                    f"bias_hh_l{layer}",
                    bias_hh,
                )

        self.reset_parameters()

    # ========================================================
    # Initialization
    # ========================================================

    def reset_parameters(self):
        """
        Match torch.nn.GRU initialization.
        """

        stdv = 1.0 / (self.hidden_size ** 0.5)

        for parameter in self.parameters():

            torch.nn.init.uniform_(
                parameter,
                -stdv,
                stdv,
            )

    # ========================================================
    # One GRU timestep
    # ========================================================

    def gru_cell(
        self,
        x_t,
        h_prev,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
    ):
        """
        x_t:
            [B, input_size]

        h_prev:
            [B, hidden_size]

        Returns:
            h_t:
            [B, hidden_size]
        """

        # ----------------------------------------------------
        # Input-to-hidden
        # ----------------------------------------------------

        gi = F.linear(
            x_t,
            weight_ih,
            bias_ih,
        )

        # ----------------------------------------------------
        # Hidden-to-hidden
        # ----------------------------------------------------

        gh = F.linear(
            h_prev,
            weight_hh,
            bias_hh,
        )

        # ----------------------------------------------------
        # Split gates
        #
        # PyTorch order:
        # r, z, n
        # ----------------------------------------------------

        i_r, i_z, i_n = gi.chunk(
            3,
            dim=1,
        )

        h_r, h_z, h_n = gh.chunk(
            3,
            dim=1,
        )

        # ----------------------------------------------------
        # Reset gate
        # ----------------------------------------------------

        reset_gate = torch.sigmoid(
            i_r + h_r
        )

        # ----------------------------------------------------
        # Update gate
        # ----------------------------------------------------

        update_gate = torch.sigmoid(
            i_z + h_z
        )

        # ----------------------------------------------------
        # New gate
        #
        # IMPORTANT:
        #
        # PyTorch:
        #
        # n = tanh(
        #     i_n
        #     +
        #     reset_gate * h_n
        # )
        #
        # h_n already contains:
        #
        # W_hn h + b_hn
        # ----------------------------------------------------

        new_gate = torch.tanh(
            i_n
            +
            reset_gate * h_n
        )

        # ----------------------------------------------------
        # Final hidden state
        #
        # PyTorch:
        #
        # h_t =
        # (1 - z) * n
        # +
        # z * h_prev
        # ----------------------------------------------------

        h_t = (
            (1.0 - update_gate)
            * new_gate
            +
            update_gate
            * h_prev
        )

        return h_t

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        x,
        h0=None,
    ):

        # x:
        # [B, T, input_size]

        batch_size = x.size(0)
        sequence_length = x.size(1)

        # ----------------------------------------------------
        # Initial hidden state
        # ----------------------------------------------------

        if h0 is None:

            h0 = torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                device=x.device,
                dtype=x.dtype,
            )

        # ----------------------------------------------------
        # Process each layer
        # ----------------------------------------------------

        layer_input = x

        final_hidden_states = []

        for layer in range(
            self.num_layers
        ):

            # Get parameters
            weight_ih = getattr(
                self,
                f"weight_ih_l{layer}",
            )

            weight_hh = getattr(
                self,
                f"weight_hh_l{layer}",
            )

            if self.bias:

                bias_ih = getattr(
                    self,
                    f"bias_ih_l{layer}",
                )

                bias_hh = getattr(
                    self,
                    f"bias_hh_l{layer}",
                )

            else:

                bias_ih = None
                bias_hh = None

            # Initial state for this layer
            h_t = h0[layer]

            timestep_outputs = []

            # ------------------------------------------------
            # Process sequence
            # ------------------------------------------------

            for t in range(
                sequence_length
            ):

                x_t = layer_input[:, t, :]

                # Observe internal GRU activations
                if getattr(self, "activation_observers", None) is not None:

                    self.activation_observers[layer]["input"].update(x_t)

                    # h_t contains the previous hidden state at this point
                    self.activation_observers[layer]["hidden"].update(h_t)

                # Internal GRU activation quantization
                if getattr(self, "activation_quantizers", None) is not None:

                    scales = self.activation_quantizers[layer]
                    bits = self.activation_bits

                    x_t = fake_quantize_activation(
                        x_t,
                        scale=scales["input"],
                        bits=bits
                    )

                    h_prev_q = fake_quantize_activation(
                        h_t,
                        scale=scales["hidden"],
                        bits=bits
                    )

                else:
                    h_prev_q = h_t

                h_t = self.gru_cell(
                    x_t=x_t,
                    h_prev=h_prev_q,

                    weight_ih=weight_ih,
                    weight_hh=weight_hh,

                    bias_ih=bias_ih,
                    bias_hh=bias_hh,
                )

                timestep_outputs.append(
                    h_t
                )

            # ------------------------------------------------
            # Stack sequence
            # ------------------------------------------------

            layer_output = torch.stack(
                timestep_outputs,
                dim=1,
            )

            # Becomes input to next layer
            layer_input = layer_output

            # Final state of this layer
            final_hidden_states.append(
                h_t
            )

        # ====================================================
        # Final output
        # ====================================================

        output = layer_input

        h_n = torch.stack(
            final_hidden_states,
            dim=0,
        )

        return output, h_n
