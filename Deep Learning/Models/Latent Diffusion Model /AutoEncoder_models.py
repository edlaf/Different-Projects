import torch.nn as nn
import models_utils as utils
from taming.modules.losses.lpips import LPIPS
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torch

class Auto_Encoder(nn.Module):
    def __init__(self, latent_channels=16, perceptual_weight=0.1):
        super(Auto_Encoder, self).__init__()
        # Encoder
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=1))
        self.resnet_1  = utils.ResNetBlock(64, 64)
        self.attn_1    = utils.AttentionBlock(64)
        self.resnet_2  = utils.ResNetBlock(64, 64)
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, latent_channels, kernel_size=3, stride=1, padding=1)
        )
        # Decoder
        self.decoder_1 = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=1))
        self.resnet_3 = utils.ResNetBlock(64, 64)
        self.attn_2 = utils.AttentionBlock(64)
        self.resnet_4 = utils.ResNetBlock(64, 64)
        self.decoder_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.perceptual_weight = perceptual_weight
        self.lpips_loss = LPIPS().eval()

    def forward(self, x):
        temb = torch.zeros(x.size(0), 512, device=x.device)
        latent_1 = self.encoder_1(x)
        latent_2 = checkpoint.checkpoint(self.resnet_1, latent_1, temb)
        latent_3 = checkpoint.checkpoint(self.attn_1, latent_2)
        latent_4 = checkpoint.checkpoint(self.resnet_2, latent_3, temb)
        latent_5 = self.encoder_2(latent_4)
        latent_6 = self.decoder_1(latent_5)
        latent_6 = checkpoint.checkpoint(self.resnet_3, latent_6, temb)
        latent_7 = checkpoint.checkpoint(self.attn_2, latent_6)
        latent_8 = checkpoint.checkpoint(self.resnet_4, latent_7, temb)
        reconstructed = self.decoder_2(latent_8)
        return reconstructed

    def loss(self, x):
        reconstructed = self.forward(x)
        nll_loss = F.l1_loss(reconstructed, x)
        perceptual_loss = self.lpips_loss(x, reconstructed)
        total_loss = nll_loss + self.perceptual_weight * perceptual_loss
        return total_loss
