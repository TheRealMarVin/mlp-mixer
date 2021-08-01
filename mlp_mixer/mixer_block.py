import torch
import torch.nn as nn
import torch.nn.functional as  F

from mlp_mixer.channel_mixer import ChannelMixer
from mlp_mixer.token_mixer import TokenMixer


class MixerBlock(nn.Module):
    def __init__(self, image_input_size, nb_channels, patch_size, hidden_size, token_hidden_size, channel_hidden_size, dropout):
        super(MixerBlock, self).__init__()

        nb_patch = int((image_input_size[0] * image_input_size[1] * nb_channels) / (patch_size[0] * patch_size[1]))

        self.pre_norm_layer = nn.LayerNorm(hidden_size)
        self.post_norm_layer = nn.LayerNorm(hidden_size)
        self.token_mixer = TokenMixer(nb_patch,
                                      token_hidden_size,
                                      dropout=dropout)
        self.channel_mixer = ChannelMixer(hidden_size,
                                          channel_hidden_size,
                                          dropout=dropout)

    def forward(self, x):
        x = self.pre_norm_layer(x)
        x = self.token_mixer(x)

        x = self.post_norm_layer(x)
        x = self.channel_mixer(x)

        return x
