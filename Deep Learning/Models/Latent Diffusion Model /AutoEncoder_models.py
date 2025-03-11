import torch.nn as nn
import models_utils as utils
from taming.modules.losses.lpips import LPIPS
import torch.nn.functional as F

class Auto_Encoder(nn.Module):
    def __init__(self, latent_channels=16, perceptual_weight = 0.1):
        super(Auto_Encoder, self).__init__()

        # Encoder : 256x256x3 → 32x32xlatent_channels
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=1))
        self.resnet_1  = utils.ResNetBlock(64, 64, temb_channels=0)
        self.attn_1    = utils.AttentionBlock(64)
        self.resnet_2  = utils.ResNetBlock(64, 64, temb_channels=0)
        self.encoder_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, latent_channels, kernel_size=3, stride=1, padding=1)
        )

        # Decoder : 32x32xlatent_channels → 256x256x3
        self.decoder_1 = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=1))
        self.resnet_3 = utils.ResNetBlock(64, 64, temb_channels=0)
        self.attn_2 = utils.AttentionBlock(64)
        self.resnet_4 = utils.ResNetBlock(64, 64, temb_channels=0)
        self.decoder_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
        self.perceptual_weight = perceptual_weight
        self.lpips_loss = LPIPS().eval()

    def forward(self, x):
        latent_1 = self.encoder_1(x)
        latent_2 = self.resnet_1(latent_1, 0)
        latent_3 = self.attn_1(latent_2)
        latent_4 = self.resnet_2(latent_3,0)
        latent_5 = self.encoder_2(latent_4)
        latent_6 = self.decoder_1(latent_5)
        latent_6 = self.resnet_3(latent_6)
        latent_7 = self.attn_2(latent_6)
        latent_8 = self.resnet_4(latent_7)
        reconstructed = self.decoder_2(latent_8)
        return reconstructed
    
    def loss(self, x):
        """
        Calcule la loss totale combinant la perte de reconstruction (nll_loss) et la perte perceptuelle (LPIPS).
        
        :param x: Batch d'images d'entrée
        :return: (total_loss, nll_loss, perceptual_loss)
        """
        reconstructed = self.forward(x)
        nll_loss = F.l1_loss(reconstructed, x)
        perceptual_loss = self.lpips_loss(x, reconstructed)
        total_loss = nll_loss + self.perceptual_weight * perceptual_loss
        return total_loss