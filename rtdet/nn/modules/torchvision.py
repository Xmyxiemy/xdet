import torch
import torch.nn as nn
from ..modules.conv import Conv, DWConv


def channel_shuffle(x, groups: int):
    # _, num_channels, _, _ = x.size()
    # channels_per_group = num_channels // groups
    return x.unflatten(dim=1, sizes=[groups, -1]).transpose(1, 2).contiguous().flatten(1, 2)


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int) -> None:
        super().__init__()

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = oup // 2
        if (self.stride == 1) and (inp != branch_features << 1):
            raise ValueError(
                f"Invalid combination of stride {stride}, inp {inp} and oup {oup} values. If stride == 1 then inp should be equal to oup // 2 << 1."
            )

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                DWConv(inp, branch_features, 3, self.stride)
                # self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                # nn.BatchNorm2d(inp),
                # nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                # nn.BatchNorm2d(branch_features),
                # nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Identity()

        self.branch2 = nn.Sequential(
            Conv(inp if (self.stride > 1) else branch_features, branch_features, 1, 1),
            # nn.Conv2d(
            #     inp if (self.stride > 1) else branch_features,
            #     branch_features,
            #     kernel_size=1,
            #     stride=1,
            #     padding=0,
            #     bias=False,
            # ),
            # nn.BatchNorm2d(branch_features),
            # nn.ReLU(inplace=True),
            DWConv(branch_features, branch_features, 3, self.stride),
            # self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            # nn.BatchNorm2d(branch_features),
            # nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(branch_features),
            # nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(
        i: int, o: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False
    ) -> nn.Conv2d:
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out
    
    @staticmethod
    def channel_shuffle(x):
        x0, x1 = x.unflatten(dim=1, sizes=[-1, 2]).chunk(2, dim=2)
        return x0.squeeze(2), x1.squeeze(2)

  
class ShuffleV2Stage(nn.Sequential):
    def __init__(self, c1, c2, reapets, same=False):
        super().__init__()
        self.add_module(name="0", module = DWConv(c1, c2) if same else InvertedResidual(c1, c2, 2))
        for name in range(1, reapets):
            self.add_module(str(name), InvertedResidual(c2, c2, 1))
