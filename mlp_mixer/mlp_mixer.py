import torch
import torch.nn as nn
import torch.nn.functional as  F

from mlp_mixer.mixer_block import MixerBlock


class MlpMixer(nn.Module):
    def __init__(self, image_input_size, nb_channels, patch_size, nb_blocks, out_size, hidden_size, dropout):
        super(MlpMixer, self).__init__()
        if type(patch_size) is tuple:
            self.patch_size = patch_size
        else:
            self.patch_size = (patch_size, patch_size)

        self.final_norm_layer = nn.LayerNorm(int(self.patch_size[0] * self.patch_size[1]))
        self.fc = nn.Linear(self.patch_size[0] * self.patch_size[1] * nb_channels, out_size)

        self.mixers = nn.ModuleList(
            [MixerBlock(image_input_size, nb_channels, patch_size, hidden_size, dropout) for _ in range(nb_blocks)]
        )

    def forward(self, x):
        x = self._create_patch(x)

        for curr_layer in self.mixers:
            x = curr_layer(x)

        x = self.final_norm_layer(x)
        x = torch.mean(x, dim=1)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _create_patch(self, x):
        shape = x.shape
        patch_count = (shape[-1] * shape[-2]) / (self.patch_size[0] * self.patch_size[1])

        # unfold channels
        x = x.data.unfold(dimension=1, size=shape[1], step=shape[1])
        # unfold width
        x = x.data.unfold(dimension=2, size=self.patch_size[0], step=self.patch_size[0])
        # unfold height
        x = x.data.unfold(dimension=3, size=self.patch_size[1], step=self.patch_size[1])

        # reshape to [batch_size, patch_count, patch_size*patch_size]
        x = x.reshape(shape[0], int(patch_count), shape[1] * self.patch_size[0] * self.patch_size[1])

        return x
