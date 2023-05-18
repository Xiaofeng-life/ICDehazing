import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import trunc_normal_


class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.network_depth = network_depth

        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.GELU(),
            nn.Conv2d(hidden_features, out_features, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1 / 4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size ** 2, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def get_relative_positions(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)

    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

    relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_positions_log = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())

    return relative_positions_log


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        relative_positions = get_relative_positions(self.window_size)
        self.register_buffer("relative_positions", relative_positions)
        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.GELU(),
            nn.Linear(256, num_heads, bias=True)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv):
        B_, N, _ = qkv.shape

        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.meta(self.relative_positions)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
        return x


class Attention(nn.Module):
    def __init__(self, network_depth, dim, num_heads, window_size, shift_size, use_attn=False):
        super().__init__()
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_heads = num_heads

        self.window_size = window_size
        self.shift_size = shift_size

        self.network_depth = network_depth
        self.use_attn = use_attn

        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
        )

        if self.use_attn:
            self.V = nn.Conv2d(dim, dim, 1)
            self.proj = nn.Conv2d(dim, dim, 1)

        if self.use_attn:
            self.QK = nn.Conv2d(dim, dim * 2, 1)
            self.attn = WindowAttention(dim, window_size, num_heads)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape

            if w_shape[0] == self.dim * 2:  # QK
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            else:
                gain = (8 * self.network_depth) ** (-1 / 4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        if shift:
            x = F.pad(x, (self.shift_size, (self.window_size - self.shift_size + mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size - self.shift_size + mod_pad_h) % self.window_size),
                      mode='reflect')
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, X):
        B, C, H, W = X.shape

        if self.use_attn:
            V = self.V(X)

        if self.use_attn:
            QK = self.QK(X)
            QKV = torch.cat([QK, V], dim=1)

            # shift
            shifted_QKV = self.check_size(QKV, self.shift_size > 0)
            Ht, Wt = shifted_QKV.shape[2:]

            # partition windows
            shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)

            qkv = window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C

            attn_windows = self.attn(qkv)

            shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)  # B H' W' C

            # reverse cyclic shift
            out = shifted_out[:, self.shift_size:(self.shift_size + H), self.shift_size:(self.shift_size + W), :]
            attn_out = out.permute(0, 3, 1, 2)

            out = self.proj(attn_out)
        else:
            out = self.conv(X)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, network_depth, dim, num_heads, mlp_ratio=4.,
                 window_size=8, shift_size=0, use_attn=True):
        super().__init__()
        self.use_attn = use_attn

        self.norm1 = nn.LayerNorm(dim) if use_attn else nn.Identity()
        self.attn = Attention(network_depth, dim, num_heads=num_heads, window_size=window_size,
                              shift_size=shift_size, use_attn=use_attn)

        self.norm2 = nn.LayerNorm(dim) if use_attn else nn.Identity()
        self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        B, C, H, W = x.size()
        identity = x
        if self.use_attn:
            x = x.view(B, C, H * W)
            x = x.transpose(1, 2)
            x = self.norm1(x)
            x = x.transpose(1, 2)
            x = x.view(B, C, H, W)
        x = self.attn(x)
        x = identity + x

        identity = x
        if self.use_attn:
            x = x.view(B, C, H * W)
            x = x.transpose(1, 2)
            x = self.norm2(x)
            x = x.transpose(1, 2)
            x = x.view(B, C, H, W)
        x = self.mlp(x)
        x = identity + x
        return x


class BasicLayer(nn.Module):
    def __init__(self, network_depth, dim, depth, num_heads, mlp_ratio=4., window_size=8, attn_ratio=0.):
        super().__init__()
        self.dim = dim
        self.depth = depth

        attn_depth = attn_ratio * depth

        use_attns = [i >= depth - attn_depth for i in range(depth)]

        self.blocks = nn.ModuleList([
            TransformerBlock(network_depth=network_depth,
                             dim=dim,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             window_size=window_size,
                             shift_size=0 if (i % 2 == 0) else window_size // 2,
                             use_attn=use_attns[i])
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


def mean_std_norm(tensor):
    mean = torch.mean(tensor)
    std = torch.std(tensor)

    tensor = (tensor - mean) / std

    tensor = tensor * 2 - 1
    return tensor


def max_min_norm(tensor):
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    return tensor


class DehazeFormerX2Y(nn.Module):
    def __init__(self, in_chans, out_chans, window_size, embed_dims, mlp_ratios, depths, num_heads, attn_ratio):
        super(DehazeFormerX2Y, self).__init__()

        # setting
        self.patch_size = 4
        self.window_size = window_size
        self.mlp_ratios = mlp_ratios

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        # backbone
        self.layer1 = BasicLayer(network_depth=sum(depths), dim=embed_dims[0], depth=depths[0],
                                 num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], window_size=window_size,
                                 attn_ratio=attn_ratio[0])

        self.patch_merge1 = PatchEmbed(patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])

        self.layer2 = BasicLayer(network_depth=sum(depths), dim=embed_dims[1], depth=depths[1],
                                 num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], window_size=window_size,
                                 attn_ratio=attn_ratio[1])

        self.patch_merge2 = PatchEmbed(patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.layer3 = BasicLayer(network_depth=sum(depths), dim=embed_dims[2], depth=depths[2],
                                 num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], window_size=window_size,
                                 attn_ratio=attn_ratio[2])

        self.layer3_add = BasicLayer(network_depth=sum(depths), dim=embed_dims[2], depth=depths[2],
                                     num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], window_size=window_size,
                                     attn_ratio=attn_ratio[2])

        self.patch_split1 = PatchUnEmbed(patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        assert embed_dims[1] == embed_dims[3]

        self.layer4 = BasicLayer(network_depth=sum(depths), dim=embed_dims[3], depth=depths[3],
                                 num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], window_size=window_size,
                                 attn_ratio=attn_ratio[3])

        self.patch_split2 = PatchUnEmbed(patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        assert embed_dims[0] == embed_dims[4]

        self.layer5_A = BasicLayer(network_depth=sum(depths), dim=embed_dims[4], depth=depths[4],
                                   num_heads=num_heads[4], mlp_ratio=mlp_ratios[4], window_size=window_size,
                                   attn_ratio=attn_ratio[4])

        self.layer5_B = BasicLayer(network_depth=sum(depths), dim=embed_dims[4], depth=depths[4],
                                   num_heads=num_heads[4], mlp_ratio=mlp_ratios[4], window_size=window_size,
                                   attn_ratio=attn_ratio[4])

        # merge non-overlapping patches into image
        self.patch_unembed_A = PatchUnEmbed(patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)
        self.patch_unembed_B = PatchUnEmbed(patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)

        self.tanh = nn.Tanh()

    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x, alpha=0.5):
        H, W = x.shape[2:]
        ori_x = self.check_image_size(x)

        x = self.patch_embed(ori_x)
        x = self.layer1(x)

        x = self.patch_merge1(x)
        x = self.layer2(x)

        x = self.patch_merge2(x)
        ori = x
        x = self.layer3(x)
        x = self.layer3_add(x)
        x = x + ori

        x = self.patch_split1(x)
        x = self.layer4(x)
        feat = self.patch_split2(x)

        #
        feat_A = self.layer5_A(feat)
        feat_A = self.patch_unembed_A(feat_A)

        feat_B = self.layer5_B(feat)
        feat_B = self.patch_unembed_B(feat_B)

        #
        part1 = alpha * feat_A
        part2 = (1 - alpha) * feat_B * ori_x
        out = part1 + part2
        out = out[:, :, :H, :W]
        out = self.tanh(out)

        if not self.training:
            return out, max_min_norm(part2[:, :, :H, :W]), \
                   max_min_norm(feat_A[:, :, :H, :W]), max_min_norm(feat_B[:, :, :H, :W])

        return out


class DehazeFormerY2X(nn.Module):
    def __init__(self, in_chans, out_chans, window_size, embed_dims, mlp_ratios, depths, num_heads, attn_ratio):
        super(DehazeFormerY2X, self).__init__()

        # setting
        self.patch_size = 4
        self.window_size = window_size
        self.mlp_ratios = mlp_ratios

        self.patch_embed = PatchEmbed(patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        #
        self.layer1 = BasicLayer(network_depth=sum(depths), dim=embed_dims[0], depth=depths[0],
                                 num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], window_size=window_size,
                                 attn_ratio=attn_ratio[0])

        self.patch_merge1 = PatchEmbed(patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])

        #
        self.layer2 = BasicLayer(network_depth=sum(depths), dim=embed_dims[1], depth=depths[1],
                                 num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], window_size=window_size,
                                 attn_ratio=attn_ratio[1])

        self.patch_merge2 = PatchEmbed(patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        #
        self.layer3 = BasicLayer(network_depth=sum(depths), dim=embed_dims[2], depth=depths[2],
                                 num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], window_size=window_size,
                                 attn_ratio=attn_ratio[2])

        self.layer3_add = BasicLayer(network_depth=sum(depths), dim=embed_dims[2], depth=depths[2],
                                     num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], window_size=window_size,
                                     attn_ratio=attn_ratio[2])

        self.patch_split1 = PatchUnEmbed(patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        assert embed_dims[1] == embed_dims[3]

        #
        self.layer4 = BasicLayer(network_depth=sum(depths), dim=embed_dims[3], depth=depths[3],
                                 num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], window_size=window_size,
                                 attn_ratio=attn_ratio[3])

        self.patch_split2 = PatchUnEmbed(patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        assert embed_dims[0] == embed_dims[4]

        #
        self.layer5 = BasicLayer(network_depth=sum(depths), dim=embed_dims[4], depth=depths[4],
                                 num_heads=num_heads[4], mlp_ratio=mlp_ratios[4], window_size=window_size,
                                 attn_ratio=attn_ratio[4])

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)

        self.tanh = nn.Tanh()

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        x = self.patch_embed(x)
        x = self.layer1(x)

        x = self.patch_merge1(x)
        x = self.layer2(x)

        x = self.patch_merge2(x)
        ori = x
        x = self.layer3(x)
        x = self.layer3_add(x)
        x = x + ori
        x = self.patch_split1(x)
        x = self.layer4(x)

        x = self.patch_split2(x)
        x = self.layer5(x)

        x = self.patch_unembed(x)

        x = x[:, :, :H, :W]

        x = self.tanh(x)
        return x


def get_FormerX2Y():
    return DehazeFormerX2Y(in_chans=3, out_chans=3, window_size=8,
                           embed_dims=[24, 48, 96, 48, 24],
                           mlp_ratios=[2., 4., 4., 4., 2.],
                           depths=[4, 8, 8, 4, 4],
                           num_heads=[2, 4, 4, 2, 1],
                           attn_ratio=[1 / 4, 1 / 2, 1 / 2, 1 / 4, 1 / 4],
                           )


def get_FormerY2X():
    return DehazeFormerY2X(in_chans=3, out_chans=3, window_size=8,
                           embed_dims=[24, 48, 96, 48, 24],
                           mlp_ratios=[2., 4., 4., 4., 2.],
                           depths=[4, 8, 8, 4, 4],
                           num_heads=[2, 4, 4, 2, 1],
                           attn_ratio=[1 / 4, 1 / 2, 1 / 2, 1 / 4, 1 / 4],
                           )


def setup_seed(seed):
    import numpy as np
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    # x = torch.randn(size=(2, 3, 256, 256)).cuda()
    # torch.save(x, "E:/CXF_Code/x.pth")
    # -0.284, 0.0034
    setup_seed(111)
    x = torch.load("E:/CXF_Code/x.pth")
    net_X2Y = get_FormerX2Y().cuda()
    net_Y2X = get_FormerY2X().cuda()
    print(torch.mean(net_X2Y(x)))
    print(torch.mean(net_Y2X(x)))
