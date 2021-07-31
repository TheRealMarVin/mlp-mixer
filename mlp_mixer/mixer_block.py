import torch
import torch.nn as nn
import torch.nn.functional as  F

from mlp_mixer.channel_mixer import ChannelMixer
from mlp_mixer.token_mixer import TokenMixer


class MixerBlock(nn.Module):
    def __init__(self, image_input_size, nb_channels, patch_size, hidden_size, dropout):
        super(MixerBlock, self).__init__()

        nb_patch = int((image_input_size[0] * image_input_size[1] * nb_channels) / (patch_size[0] * patch_size[1]))
        patch_area = int(patch_size[0] * patch_size[1] * nb_channels)
        self.pre_norm_layer = nn.LayerNorm(hidden_size)
        self.post_norm_layer = nn.LayerNorm(hidden_size)
        self.token_mixer = TokenMixer(nb_patch,
                                      hidden_size * 4,
                                      dropout=dropout)
        self.channel_mixer = ChannelMixer(hidden_size,
                                          hidden_size * 4,
                                          dropout=dropout)

    def forward(self, x):
        x = self.pre_norm_layer(x)
        x = self.token_mixer(x)

        x = self.post_norm_layer(x)
        x = self.channel_mixer(x)

        return x
