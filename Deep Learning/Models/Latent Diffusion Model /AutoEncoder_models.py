import torch.nn as nn
import models_utils as utils
from taming.modules.losses.lpips import LPIPS
import torch.nn.functional as F

class Auto_Encoder(nn.Module):
    def __init__(self, latent_channels=16, perceptual_weight = 0.1):
        super(Auto_Encoder, self).__init__()

        # Encoder : Réduit la taille 256x256x3 → 32x32xlatent_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.LeakyReLU(),
            utils.ResNetBlock(64, 64, temb_channels=0),
            utils.AttentionBlock(64),
            utils.ResNetBlock(64, 64, temb_channels=0),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.LeakyReLU(),
#            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 32x32
#            nn.ReLU(),
            nn.Conv2d(128, latent_channels, kernel_size=3, stride=1, padding=1)  # 32x32xlatent_channels
        )

        # Decoder : Reconstruit l'image 32x32xlatent_channels → 256x256x3
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.LeakyReLU(),
            utils.ResNetBlock(64, 64, temb_channels=0),
            utils.AttentionBlock(64),
            utils.ResNetBlock(64, 64, temb_channels=0),
#            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 256x256
#            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),  # Output 3 channels
            nn.Sigmoid()  # Normalisation entre 0 et 1
        )
        
        self.perceptual_weight = perceptual_weight
        self.lpips_loss = LPIPS().eval()

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
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