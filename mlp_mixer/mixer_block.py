import torch
import torch.nn as nn
import torch.nn.functional as  F

from mlp_mixer.channel_mixer import ChannelMixer
from mlp_mixer.token_mixer import TokenMixer


class MixerBlock(nn.Module):
    def __init__(self, image_input_size, nb_channels, patch_size, hidden_size, dropout):
        super(MixerBlock, self).__init__()
        self.pre_norm_layer = nn.LayerNorm(int((patch_size * patch_size)))
        self.post_norm_layer = nn.LayerNorm(int((patch_size * patch_size)))
        self.token_mixer = TokenMixer(int((image_input_size * image_input_size) / (patch_size * patch_size)),
                                      hidden_size,
                                      nb_channels,
                                      dropout=dropout)
        self.channel_mixer = ChannelMixer(patch_size * patch_size * nb_channels,
                                          hidden_size,
                                          nb_channels,
                                          dropout=dropout)

    def forward(self, x):
        x = self.pre_norm_layer(x)
        x = self.token_mixer(x)

        x = self.post_norm_layer(x)
        x = self.channel_mixer(x)

        return x
