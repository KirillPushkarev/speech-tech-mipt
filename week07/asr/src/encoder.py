import torch
from torch import nn
from torch.nn import ReLU


def get_conv_and_norm_layers(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        separable: bool = False,
):
    if separable:
        layers = [
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                groups=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding
            ),
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                dilation=1,
                padding=0,
            ),
        ]
    else:
        layers = [
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding
            )
        ]

    layers.append(nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1))
    return layers


def get_activation_and_dropout_layers(activation=ReLU, drop_prob=0.2):
    layers = [activation(), nn.Dropout(p=drop_prob)]
    return layers


def get_same_padding(kernel_size: int, stride: int, dilation: int) -> int:
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    return (dilation * (kernel_size - 1)) // 2


class QuartzNetBlock(torch.nn.Module):
    def __init__(
            self,
            feat_in: int,
            filters: int,
            repeat: int,
            kernel_size: int,
            stride: int,
            dilation: int,
            residual: bool,
            separable: bool,
            dropout: float,
    ):
        super().__init__()

        padding = get_same_padding(kernel_size, stride, dilation)

        sub_blocks = []
        sub_block_input_channels = feat_in
        for block in range(repeat - 1):
            sub_blocks.extend(
                get_conv_and_norm_layers(
                    sub_block_input_channels,
                    filters,
                    kernel_size,
                    stride,
                    dilation,
                    padding,
                    separable=separable
                )
            )
            sub_blocks.extend(
                get_activation_and_dropout_layers(activation=ReLU)
            )
            sub_block_input_channels = filters

        sub_blocks.extend(
            get_conv_and_norm_layers(
                sub_block_input_channels,
                filters,
                kernel_size,
                stride,
                dilation,
                padding,
                separable=separable
            )
        )

        self.mainline = nn.Sequential(*sub_blocks)

        if residual:
            self.residual = nn.Sequential(
                *get_conv_and_norm_layers(
                    feat_in,
                    filters,
                    kernel_size=1,
                    stride=1,
                    dilation=1,
                    padding=0,
                    separable=False
                )
            )
        else:
            self.residual = None

        self.out = nn.Sequential(*get_activation_and_dropout_layers(activation=ReLU, drop_prob=dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual:
            return self.out(self.mainline(x) + self.residual(x))

        return self.out(self.mainline(x))


class QuartzNet(nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.stride_val = 1

        layers = []
        feat_in = conf.feat_in
        for block in conf.blocks:
            layers.append(QuartzNetBlock(feat_in, **block))
            self.stride_val *= block.stride ** block.repeat
            feat_in = block.filters

        self.layers = nn.Sequential(*layers)

    def forward(
            self, features: torch.Tensor, features_length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.layers(features)
        encoded_len = (
                torch.div(features_length - 1, self.stride_val, rounding_mode="trunc") + 1
        )

        return encoded, encoded_len
