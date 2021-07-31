# mlp-mixer

This is my attempt to implement the MLP-mixer model as described in this paper: https://arxiv.org/pdf/2105.01601v1.pdf

Based on the code of what seems to be the original implementation:  https://github.com/google-research/vision_transformer/blob/master/vit_jax/models_mixer.py

## Usage
To install requirement for this project : 
conda env create -f environment.yml

Note: I think you only need Pytorch and maybe numpy to run this code. So I suggest you use your own environment and just install pytorch for GPU.

```
model = MlpMixer(image_input_size=img_size,
                     nb_channels=1,
                     patch_size=4,
                     nb_blocks=4,
                     out_size=10,
                     hidden_size=256,
                     dropout=0.1)
```

image_input_size and patch_size can be a tupple for 2D shape.
nb_channels: is the number of the channel for the input.
nb_blocks: is the number of mixer block in the model.
out_size: The number of classes for the output of the model.
hidden_size: The hidden size of the mixer block.
dropout: The amount of dropout to use.

## Install
Let's make it works
