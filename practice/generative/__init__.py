"""Generative models: autoencoders, VAEs, VQ-VAE, GANs, DDPM and flow matching (PyTorch).

All models are tiny MLPs on 2-D toy data so every test runs on CPU in seconds; the
maths and the training steps are exactly the ones used at scale.
"""

from . import autoencoder, ddpm, flow_matching, gan, toy_data, vae, vqvae

__all__ = ["autoencoder", "ddpm", "flow_matching", "gan", "toy_data", "vae", "vqvae"]
