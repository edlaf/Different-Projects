import torch.nn as nn

class Auto_Encoder(nn.Module):
    def __init__(self, latent_channels=16):
        super(Auto_Encoder, self).__init__()

        # Encoder : Réduit la taille 256x256x3 → 32x32xlatent_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
#            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 32x32
#            nn.ReLU(),
            nn.Conv2d(128, latent_channels, kernel_size=3, stride=1, padding=1)  # 32x32xlatent_channels
        )

        # Decoder : Reconstruit l'image 32x32xlatent_channels → 256x256x3
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.ReLU(),
#            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 256x256
#            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),  # Output 3 channels
            nn.Sigmoid()  # Normalisation entre 0 et 1
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed