import torch
import torch.nn.functional as F
from dataclasses import dataclass, field


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
    def __init__(self, C):
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
        self.comp_glinear = GLinear(2*C*80, C*4, 8)

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

        self.encoder = Encoder(C)
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
