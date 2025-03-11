import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, TensorDataset
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def load_data(path,batch_size = 128):
    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = random_split(dataset, [train_size, test_size])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    return trainloader, testloader, dataloader

def show(images):

    images = images.cpu().detach()
    _, axes = plt.subplots(5, 5, figsize=(6, 6))

    for i, ax in enumerate(axes.flat):
        img = images[i].permute(1, 2, 0).numpy()
        ax.imshow(img,)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def show_images_side_by_side(images, reconstructed_images, nrow=8, title="Original vs Reconstructed"):
    """
    Affiche les images originales et leurs reconstructions côte à côte.

    :param images: Tenseur Pytorch des images originales de taille (N, C, H, W)
    :param reconstructed_images: Tenseur Pytorch des images reconstruites de taille (N, C, H, W)
    :param nrow: Nombre d'images par ligne
    :param title: Titre de la figure
    """
    if images.shape != reconstructed_images.shape:
        raise ValueError("Les tenseurs d'images doivent avoir la même forme.")

    stacked_images = torch.cat((images, reconstructed_images), dim=0)
    grid = torchvision.utils.make_grid(stacked_images, nrow=nrow, normalize=True, scale_each=True)

    plt.figure(figsize=(12, 6))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.title(title)
    plt.show()
    
    

def train(model, trainloader, optimizer, testloader, device, nb_epochs=10, display_interval=5):
    nb_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters of the model {nb_params}")

    test_images, _ = next(iter(testloader))
    test_images = test_images.to(device)

    for epoch in range(nb_epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{nb_epochs}")
        i = 0

        for images, _ in progress_bar:
            images = images.to(device)
            loss = model.loss(images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item()
            progress_bar.set_postfix(loss=total_loss / (i + 1))
            i += 1

        model.eval()
        with torch.no_grad():
            reconstructed_test = model(test_images)

        if (epoch + 1) % display_interval == 0:
            images_cpu = test_images.cpu()
            reconstructed_cpu = reconstructed_test.cpu()
            show_images_side_by_side(images_cpu[:24], reconstructed_cpu[:24])

    print("Training Over.")
    return model

    
def test(model, criterion, testloader, device):
    total_test_loss = 0.0
    model.eval()
    criterion = model.loss_2
    with torch.no_grad():
        for images, _ in tqdm(testloader):
            images = images.to(device)
            reconstructed_images = model(images)
            loss = criterion(images, reconstructed_images)
            total_test_loss += loss

    avg_test_loss = total_test_loss / len(testloader)
    print(f"Test Loss: {avg_test_loss:.7f}")
    
def Encode_data(autoencoder, device, dataloader):
    autoencoder.eval()

    encoded_images = []
    labels_list = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            encoded = autoencoder.encoder(images)
            encoded_images.append(encoded)
            labels_list.append(labels)

    encoded_images = torch.cat(encoded_images, dim=0)
    labels_list = torch.cat(labels_list, dim=0)
    encoded_dataset = TensorDataset(encoded_images, labels_list)
    encoded_dataloader = DataLoader(encoded_dataset, batch_size=128, shuffle=True)

    print(f"Nombre total d'images encodées : {len(encoded_dataset)}")
    return encoded_dataloader