from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp

from .registry import register_model


__all__ = ['EfficientViT']


@torch.jit.script
def hard_swish(x: torch.Tensor):
    y = F.relu6(x + 3) / 6

    return x * y


class HardSwish(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return hard_swish(x)


class ConvBNAct(nn.Module):
    def __init__(self, *args, act=nn.ReLU, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(*args, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(self.conv.out_channels)
        self.act = act()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


conv_swish = partial(ConvBNAct, act=HardSwish)


class FusedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, expansion=4):
        super().__init__()

        self.body = nn.Sequential(
            ConvBNAct(in_channels, in_channels * expansion, kernel_size, padding=kernel_size // 2, stride=stride),
            nn.Conv2d(in_channels * expansion, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = None

    def forward(self, x):
        res = self.body(x)
        if self.shortcut is not None:
            res = res + self.shortcut(x)
        return res


class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.branch_a = ConvBNAct(in_channels, out_channels, 3, stride=2, padding=1)
        self.branch_b = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBNAct(in_channels, out_channels, 1)
        )

    def forward(self, x):
        a = self.branch_a(x)
        b = self.branch_b(x)

        return a + b


class Fused_IRB_Group(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, expansion=4):
        super().__init__()

        self.channels = out_channels

        self.head = FusedMBConv(in_channels, out_channels, 3, stride=2, expansion=expansion)

        self.tail = nn.Sequential(*[
            FusedMBConv(out_channels, out_channels, 3, expansion=expansion)
            for _ in range(1, num_blocks)
        ])

    def forward(self, x):
        x = self.head(x)
        x = self.tail(x)
        return x


class LinearAttentionBlock(nn.Module):
    def __init__(self, channels, heads=12, expansion=4, key_dim=16):
        super().__init__()

        self.heads = heads
        self.channels = channels
        self.key_dim = key_dim
        self.proj_slice_size = heads * key_dim

        self.projection = ConvBNAct(channels + 2, 3 * self.proj_slice_size, 1)

        self.value_proj_2 = ConvBNAct(heads * key_dim, channels, 1)

        self.ffn = nn.Sequential(
            conv_swish(channels, channels, 5, padding=2, groups=channels),
            conv_swish(channels, channels * expansion, 1),
            nn.Conv2d(channels * expansion, channels, 1),
            nn.BatchNorm2d(channels),
        )

        self.rz_attn = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)
        self.rz_ffn = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)
        self.embed_strength = nn.Parameter(torch.tensor(-10, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        B, _, H, W = x.shape

        x_pos = torch.arange(0, W, dtype=x.dtype, device=x.device) / 1000
        y_pos = torch.arange(0, H, dtype=x.dtype, device=x.device) / 1000
        x_pos, y_pos = torch.meshgrid(x_pos, y_pos, indexing='xy')
        pos_emb = torch.stack([x_pos, y_pos], dim=0).unsqueeze(0).expand(B, -1, -1, -1)

        # This will force the model to learn relative offsets since the absolute positions
        # are never consistent, but the relative difference between two positions is constant.
        rnd_offset = torch.rand(B, 2, 1, 1, dtype=x.dtype, device=x.device) * 6 - 3
        pos_emb = pos_emb + rnd_offset
        embed_strength = self.embed_strength.exp()
        pos_emb = embed_strength * pos_emb

        x_emb = torch.cat((x, pos_emb), dim=1)

        proj = self.projection(x_emb)

        with amp.autocast(enabled=False):
            proj = proj.float()
            get_slice = lambda offset: proj[:, offset * self.proj_slice_size : (1 + offset) * self.proj_slice_size].reshape(B * self.heads, self.key_dim, H * W).transpose(1, 2)

            # <batch * heads> x <positions> x <channels> === BNC
            q = get_slice(0)
            k = get_slice(1)
            v = get_slice(2)

            # BCC
            ktv = torch.matmul(k.transpose(1, 2), v)

            # BNC
            numer = torch.matmul(q, ktv)

            # B1C
            k_sum = torch.sum(k, dim=1, keepdim=True)

            # BN1
            denom = torch.matmul(q, k_sum.transpose(1, 2))

            # BNC
            output = numer / denom.clamp_min(1e-4)

            # BCN
            output = output.transpose(1, 2).contiguous()

            output = output.reshape(B, -1, H, W)

        output = self.value_proj_2(output)

        y = x + self.rz_attn * output
        # y = x

        z = y + self.rz_ffn * self.ffn(y)

        return z


class LATGroup(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, heads=12, expansion=4, key_dim=16):
        super().__init__()

        self.channels = out_channels

        self.mbconv = FusedMBConv(in_channels, out_channels, 3, stride=2, expansion=expansion)
        self.lats = nn.Sequential(*[
            LinearAttentionBlock(out_channels, heads, expansion, key_dim)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = self.mbconv(x)
        x = self.lats(x)
        return x


class EfficientViT(nn.Module):
    def __init__(self, num_classes=1000, **kwargs):
        super().__init__()

        self.num_classes = num_classes

        # C0=16, C1=24, C2=48, C3=96, C4=192; L1=2, L2=3, L3=5, L4=2

        # In linear attention, the key/value dimension is 16, while the number of heads is 12/24 in stage 3/4.

        # To highlight the efficient backbone itself, we keep the hyper-parameters simple using the same expand
        # ratio e for MBConv [22] and FFN (e = 4), the same kernel size k for all depthwise convolution (k =
        # 5 except the input stem), and the same activation function (hard swish [48]) for all layers.

        expansion = 4

        self.trunk = nn.Sequential(
            ConvBNAct(3, 16, 3, padding=1),
            DSConv(16, 32),
        )

        self.stages = nn.ModuleList([
            Fused_IRB_Group(32, 32, num_blocks=2, expansion=expansion), # C1
            Fused_IRB_Group(32, 64, num_blocks=3, expansion=expansion), # C2
            LATGroup(64, 96, num_blocks=5, heads=12, expansion=expansion), # C3
            LATGroup(96, 192, num_blocks=2, heads=24, expansion=expansion), # C4
        ])

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.final = nn.Linear(192, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # m.weight.data.mul_(0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.trunk(x)

        for stage in self.stages:
            x = stage(x)

        y = self.pool(x)
        y = self.final(y.flatten(1))

        return y


@register_model
def efficient_vit_tiny(pretrained=False, **kwargs):
    return EfficientViT(**kwargs)
