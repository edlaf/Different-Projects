import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import models_utils as utils
#from taming.modules.losses.lpips import LPIPS

class Auto_Encoder(nn.Module):
    def __init__(self, latent_channels=3, perceptual_weight=0.1):
        super(Auto_Encoder, self).__init__()
        
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=1)
        )
        self.resnet_1 = utils.ResNetBlock(64, 64)
        self.down_sample = utils.Downsample(64)
        self.down_to_16_enc = nn.Conv2d(64, 64, kernel_size=4, stride=4, padding=0)
        self.attn_1 = utils.AttentionBlock(64)
        self.resnet_2 = utils.ResNetBlock(64, 64)
        self.up_to_32_enc = utils.Upsample(64)
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, latent_channels, kernel_size=3, stride=1, padding=1)
        )

        self.decoder_1 = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=1)   # conserve 64x64
        )
        self.down_to_16_dec = nn.Conv2d(64, 64, kernel_size=4, stride=4, padding=0)
        self.resnet_3 = utils.ResNetBlock(64, 64)
        self.attn_2 = utils.AttentionBlock(64)
        self.up_to_32_dec = utils.Upsample(64)
        self.resnet_4 = utils.ResNetBlock(64, 64)
        self.up_sample_final = utils.Upsample(64)
        self.final_upsample = nn.Upsample(scale_factor=4, mode='nearest')
        self.decoder_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
        self.perceptual_weight = perceptual_weight
        #self.lpips_loss = LPIPS().eval()

    def forward(self, x):
        temb = torch.zeros(x.size(0), 512, device=x.device)
        
        latent_1 = self.encoder_1(x)
        latent_1 = self.resnet_1(latent_1, temb)
        latent_2 = self.down_sample(latent_1)
        latent_2_small = self.down_to_16_enc(latent_2)
        latent_3 = self.attn_1(latent_2_small)
        latent_4 = self.resnet_2(latent_3, temb)
        latent_4_big = self.up_to_32_enc(latent_4)
        latent_5 = self.encoder_2(latent_4_big)
        
        latent_6 = self.decoder_1(latent_5)
        latent_6_small = self.down_to_16_dec(latent_6)
        latent_7 = self.resnet_3(latent_6_small, temb)
        latent_8 = self.attn_2(latent_7)
        latent_8_big = self.up_to_32_dec(latent_8)
        latent_9 = self.resnet_4(latent_8_big, temb)
        latent_10 = self.up_sample_final(latent_9)
        decoded = self.decoder_2(latent_10)
        reconstructed = self.final_upsample(decoded)
        return reconstructed

    def encoder(self, x):
        temb = torch.zeros(x.size(0), 512, device=x.device)
        latent_1 = self.encoder_1(x)
        latent_1 = self.resnet_1(latent_1, temb)
        latent_2 = self.down_sample(latent_1)
        latent_2_small = self.down_to_16_enc(latent_2)
        latent_3 = self.attn_1(latent_2_small)
        latent_4 = self.resnet_2(latent_3, temb)
        latent_4_big = self.up_to_32_enc(latent_4)
        latent_5 = self.encoder_2(latent_4_big)
        
        return latent_5
    
    def decoder(self,x):
        temb = torch.zeros(x.size(0), 512, device=x.device)
        latent_6 = self.decoder_1(x)
        latent_6_small = self.down_to_16_dec(latent_6)
        latent_7 = checkpoint.checkpoint(self.resnet_3, latent_6_small, temb)
        latent_8 = checkpoint.checkpoint(self.attn_2, latent_7)
        latent_8_big = self.up_to_32_dec(latent_8)
        latent_9 = checkpoint.checkpoint(self.resnet_4, latent_8_big, temb)
        latent_10 = self.up_sample_final(latent_9)
        decoded = self.decoder_2(latent_10)
        reconstructed = self.final_upsample(decoded)
        return reconstructed
    def loss(self, x):
        reconstructed = self.forward(x)
        nll_loss = F.l1_loss(reconstructed, x)
        #perceptual_loss = self.lpips_loss(x, reconstructed).mean()
        total_loss = nll_loss #+ self.perceptual_weight * perceptual_loss
        return total_loss
    
    def loss_2(self, x, reconstructed):
        nll_loss = F.l1_loss(reconstructed, x)
        #perceptual_loss = self.lpips_loss(x, reconstructed).mean()
        total_loss = nll_loss# + self.perceptual_weight * perceptual_loss
        return total_loss

