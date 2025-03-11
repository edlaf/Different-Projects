import torch.nn as nn
import models_utils as utils
from taming.modules.losses.lpips import LPIPS
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torch
# On suppose que dans models_utils, vos classes Downsample, Upsample,
# ResNetBlock et AttentionBlock sont définies.

class Auto_Encoder(nn.Module):
    def __init__(self, latent_channels=16, perceptual_weight=0.1):
        super(Auto_Encoder, self).__init__()
        ### ENCODEUR
        # Partie initiale de l'encodeur (extraction de features)
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # ex: 256x256 -> 128x128
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=1)
        )
        self.resnet_1 = utils.ResNetBlock(64, 64)  # opère en 128x128

        # On souhaite réduire la résolution à 16x16 AVANT l'attention et le bloc ResNet.
        # D'abord, si besoin, on redescend de 128x128 à 64x64
        self.down_sample = utils.Downsample(64)  # 128x128 -> 64x64
        
        # Ensuite, on réduit fortement pour atteindre 16x16
        # Ici on utilise une convolution avec kernel=4, stride=4 (sans padding) :
        self.down_to_16_enc = nn.Conv2d(64, 64, kernel_size=4, stride=4, padding=0)  # 64x64 -> 16x16
        
        # Blocs opérant sur une résolution réduite (16x16)
        self.attn_1 = utils.AttentionBlock(64)   # sur 16x16
        self.resnet_2 = utils.ResNetBlock(64, 64)   # sur 16x16
        
        # Remonter la résolution à 32x32 pour continuer l'encodage
        self.up_to_32_enc = utils.Upsample(64)  # 16x16 -> 32x32
        
        # Suite de l'encodeur pour obtenir la représentation latente finale
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),      # conserve 32x32
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.LeakyReLU(),
            nn.Conv2d(128, latent_channels, kernel_size=3, stride=1, padding=1)  # reste en 16x16
        )
        
        ### DÉCODEUR
        # On part d'une représentation latente en 16x16
        self.decoder_1 = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),                # 32x32 -> 64x64
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=1)   # conserve 64x64
        )
        
        # Pour imiter la même stratégie que l'encodeur, on redescend la résolution avant d'appliquer l'attention.
        self.down_to_16_dec = nn.Conv2d(64, 64, kernel_size=4, stride=4, padding=0)  # 64x64 -> 16x16
        
        # Blocs sur résolution réduite dans le décodeur
        self.resnet_3 = utils.ResNetBlock(64, 64)  # sur 16x16
        self.attn_2 = utils.AttentionBlock(64)      # sur 16x16
        
        # Remonter la résolution à 32x32 après ces blocs
        self.up_to_32_dec = utils.Upsample(64)  # 16x16 -> 32x32
        
        # Ensuite, poursuivre le décodage
        self.resnet_4 = utils.ResNetBlock(64, 64)  # opère sur 32x32
        # Si besoin, on peut ajouter une étape d'upsampling supplémentaire pour atteindre la taille finale.
        # Ici, on utilise un upsample pour passer de 32x32 à 64x64 (par exemple)
        self.up_sample_final = utils.Upsample(64)  # 32x32 -> 64x64
        
        # Dernière couche pour générer l'image finale (ici on suppose une sortie en 3 canaux)
        self.decoder_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
        self.perceptual_weight = perceptual_weight
        self.lpips_loss = LPIPS().eval()

    def forward(self, x):
        # On crée un embedding temporel nul (si non utilisé)
        temb = torch.zeros(x.size(0), 512, device=x.device)
        
        # ----- ENCODEUR -----
        latent_1 = self.encoder_1(x)                               # Ex: (B, 64, 128, 128)
        latent_1 = checkpoint.checkpoint(self.resnet_1, latent_1, temb)  # (B, 64, 128, 128)
        latent_2 = self.down_sample(latent_1)                      # (B, 64, 64, 64)
        latent_2_small = self.down_to_16_enc(latent_2)             # (B, 64, 16, 16)
        latent_3 = checkpoint.checkpoint(self.attn_1, latent_2_small)  # (B, 64, 16, 16)
        latent_4 = checkpoint.checkpoint(self.resnet_2, latent_3, temb)  # (B, 64, 16, 16)
        latent_4_big = self.up_to_32_enc(latent_4)                 # (B, 64, 32, 32)
        latent_5 = self.encoder_2(latent_4_big)                    # (B, latent_channels, 16, 16)
        
        # ----- DÉCODEUR -----
        latent_6 = self.decoder_1(latent_5)                        # (B, 64, 64, 64)
        latent_6_small = self.down_to_16_dec(latent_6)             # (B, 64, 16, 16)
        latent_7 = checkpoint.checkpoint(self.resnet_3, latent_6_small, temb)  # (B, 64, 16, 16)
        latent_8 = checkpoint.checkpoint(self.attn_2, latent_7)    # (B, 64, 16, 16)
        latent_8_big = self.up_to_32_dec(latent_8)                 # (B, 64, 32, 32)
        latent_9 = checkpoint.checkpoint(self.resnet_4, latent_8_big, temb)  # (B, 64, 32, 32)
        latent_10 = self.up_sample_final(latent_9)                 # (B, 64, 64, 64)
        reconstructed = self.decoder_2(latent_10)                # (B, 3, 64, 64)
        return reconstructed

    def loss(self, x):
        reconstructed = self.forward(x)
        nll_loss = F.l1_loss(reconstructed, x)
        perceptual_loss = self.lpips_loss(x, reconstructed)
        total_loss = nll_loss + self.perceptual_weight * perceptual_loss
        return total_loss
