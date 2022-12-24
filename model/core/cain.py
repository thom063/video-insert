

import torch.nn as nn

from .common import *


class Encoder(nn.Module):
    def __init__(self, in_channels=3, depth=3, input_size=None):
        super(Encoder, self).__init__()

        # Shuffle pixels to expand in channel dimension
        # shuffler_list = [PixelShuffle(0.5) for i in range(depth)]
        # self.shuffler = nn.Sequential(*shuffler_list)
        self.shuffler = PixelShuffle(1 / 2**depth, input_size)

        relu = nn.LeakyReLU(0.2, True)
        
        # FF_RCAN or FF_Resblocks
        self.interpolate = Interpolation(5, 12, in_channels * (4**depth), act=relu)
        
    def forward(self, x1, x2):
        """
        Encoder: Shuffle-spread --> Feature Fusion --> Return fused features
        """
        feats1 = self.shuffler(x1)
        feats2 = self.shuffler(x2)

        feats = self.interpolate(feats1, feats2)

        return feats


class Decoder(nn.Module):
    def __init__(self, depth=3, input_size=None):
        super(Decoder, self).__init__()

        # shuffler_list = [PixelShuffle(2) for i in range(depth)]
        # self.shuffler = nn.Sequential(*shuffler_list)
        self.shuffler = PixelShuffle(2**depth, input_size)

    def forward(self, feats):
        out = self.shuffler(feats)
        return out


class CAIN(nn.Module):
    def __init__(self, depth=3, input_size=None):
        super(CAIN, self).__init__()
        self.input_size = input_size
        self.encoder = Encoder(in_channels=3, depth=depth, input_size=self.input_size)
        encode_feat = None if input_size is None else (input_size[0], input_size[1]*64, input_size[2]//8, input_size[3]//8)
        self.decoder = Decoder(depth=depth, input_size=encode_feat)

    def forward(self, x1, x2):
        x1, m1 = sub_mean(x1)
        x2, m2 = sub_mean(x2)

        feats = self.encoder(x1, x2)
        out = self.decoder(feats)

        mi = (m1 + m2) / 2
        out += mi

        return out, feats
