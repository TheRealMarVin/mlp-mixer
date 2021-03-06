import torch
import torch.nn as nn
import torch.nn.functional as  F

from mlp_mixer.mixer_block import MixerBlock


class MlpMixer(nn.Module):
    def __init__(self, image_input_size, nb_channels, patch_size,
                 nb_blocks, out_size, hidden_size, dropout,
                 token_hidden_size=None, channel_hidden_size=None):
        super(MlpMixer, self).__init__()

        if token_hidden_size is None:
            token_hidden_size = hidden_size
        if channel_hidden_size is None:
            channel_hidden_size = hidden_size

        self.patch_size = self._make_tuple(patch_size)
        self.image_input_size = self._make_tuple(image_input_size)

        self.final_norm_layer = nn.LayerNorm(hidden_size)
        self.patch_fc = nn.Linear(int(self.patch_size[0] * self.patch_size[1]), hidden_size)
        self.fc = nn.Linear(hidden_size, out_size)

        self.mixers = nn.ModuleList(
            [MixerBlock(self.image_input_size, nb_channels, self.patch_size, hidden_size, token_hidden_size, channel_hidden_size, dropout) for _ in range(nb_blocks)]
        )

    def _make_tuple(self, val):
        if type(val) is tuple:
            res = val
        else:
            res = (val, val)

        return res

    def forward(self, x):
        x = self._create_patch(x)
        x = self.patch_fc(x)

        for curr_layer in self.mixers:
            x = curr_layer(x)

        x = self.final_norm_layer(x)
        x = torch.mean(x, dim=1)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _create_patch(self, x):
        shape = x.shape
        patch_count = (shape[-1] * shape[-2]  * shape[-3]) / (self.patch_size[0] * self.patch_size[1])

        # unfold channels
        x = x.data.unfold(dimension=1, size=shape[1], step=shape[1])
        # unfold width
        x = x.data.unfold(dimension=2, size=self.patch_size[0], step=self.patch_size[0])
        # unfold height
        x = x.data.unfold(dimension=3, size=self.patch_size[1], step=self.patch_size[1])

        # reshape to [batch_size, patch_count, patch_size*patch_size]
        x = x.reshape(shape[0], int(patch_count), self.patch_size[0] * self.patch_size[1])

        return x
