import torch
import torch.nn as nn
from tqdm.auto import tqdm
import AutoEncoder as AutoEncoder


class DiffusionModel:

    def __init__(self, T: int, model: nn.Module, device: str):
        self.T = T
        self.function_approximator = model.to(device)
        self.device = device

        self.beta = torch.linspace(1e-4, 0.02, T).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def training(self,encoded_dataloader, optimizer, autoencoder, nb_epochs=50):

        for epoch in range(nb_epochs):

            self.function_approximator.train()

            total_loss = 0.0
            i = 0
            progress_bar = tqdm(encoded_dataloader, desc=f'Epoch {epoch + 1}/{nb_epochs}')

            for x0, _ in progress_bar:

                x0 = x0.to(self.device)
                batch_size = x0.size(0)
                t = torch.randint(1, self.T + 1, (batch_size,), device=self.device,
                                dtype=torch.long)
                eps = torch.randn_like(x0)

                # Take one gradient descent step
                alpha_bar_t = self.alpha_bar[t - 1].unsqueeze(-1).unsqueeze(
                    -1).unsqueeze(-1)
                eps_predicted = self.function_approximator(torch.sqrt(
                    alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps, t - 1)
                loss = nn.functional.mse_loss(eps, eps_predicted)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                progress_bar.set_postfix({"Loss": total_loss / (i + 1)})
                i += 1

            self.function_approximator.eval()
            with torch.no_grad():
                generated_images = self.sampling(n_samples=25, image_channels=2, img_size=(32, 32))
                decoded_generated_images = autoencoder.decoder(generated_images)
                AutoEncoder.show(decoded_generated_images)

        return loss.item()

    @torch.no_grad()
    def sampling(self, n_samples=25, image_channels=1, img_size=(32, 32),
                use_tqdm=True):
        """
        Algorithm 2 in Denoising Diffusion Probabilistic Models
        """

        x = torch.randn((n_samples, image_channels, img_size[0], img_size[1]),
                        device=self.device)
        progress_bar = tqdm if use_tqdm else lambda x: x
        for t in progress_bar(range(self.T, 0, -1)):
            z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
            t = torch.ones(n_samples, dtype=torch.long, device=self.device) * t

            beta_t = self.beta[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            alpha_t = self.alpha[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            alpha_bar_t = self.alpha_bar[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            mean = 1 / torch.sqrt(alpha_t) * (x - ((1 - alpha_t) / torch.sqrt(
                1 - alpha_bar_t)) * self.function_approximator(x, t - 1))
            sigma = torch.sqrt(beta_t)
            x = mean + sigma * z
        return x