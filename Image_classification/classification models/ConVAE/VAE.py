import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom dataset
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.images = [img for img in os.listdir(data_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.images[idx])
        image = Image.open(img_path).convert('L')  # Convert image to grayscale
        if self.transform:
            image = self.transform(image)
        return image

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256,256)),  # Resize to the input size expected by your model
    transforms.ToTensor(),

])

# Dataset and DataLoader
data_dir = '/Users/jadeja19/Documents/Hiwi iktd/data set/allgemein1-5/02corrected'
dataset = CustomDataset(data_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# VAE model
class VAE(nn.Module):

    def __init__(self, z_dim):
        super(VAE, self).__init__()

        self.z_dim = z_dim

    #encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.conv_out_size = self._get_conv_out_size((1, 256, 256))
        self.mu = nn.Sequential(
            nn.Linear(self.conv_out_size, z_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2)

        )
        self.log_var = nn.Sequential(
            nn.Linear(self.conv_out_size, z_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        # decoder
        self.decoder_linear = nn.Sequential(
            nn.Linear(z_dim, self.conv_out_size),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, 2, stride=2, padding=0),
            nn.Sigmoid()
        )
        # self.decoder_conv = nn.Sequential(
        # nn.UpsamplingNearest2d(scale_factor=2),
        # nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
        # nn.BatchNorm2d(64),
        # nn.LeakyReLU(),
        # nn.UpsamplingNearest2d(scale_factor=2),
        # nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
        # nn.BatchNorm2d(64),
        # nn.LeakyReLU(),
        # nn.UpsamplingNearest2d(scale_factor=2),
        # nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
        # nn.BatchNorm2d(32),
        # nn.LeakyReLU(),
        # nn.UpsamplingNearest2d(scale_factor=2),
        # nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1),
        # nn.Sigmoid()
        # )

    def _get_conv_out_size(self, shape):
        with torch.no_grad():
            o = self.encoder_conv(torch.zeros(1, *shape))
            return int(np.prod(o.size()))




    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoder
        x_conv = self.encoder_conv(x)
        x_conv_flat = x_conv.view(x_conv.size(0), -1)
        mu = self.mu(x_conv_flat)
        logvar = self.log_var(x_conv_flat)
        z = self.reparameterize(mu, logvar)

        # Decoder
        x_lin = self.decoder_linear(z)
        x_lin_view = x_lin.view(x.size(0), 64, 4, 4)  # Adjust shape to match decoder input
        x_recon = self.decoder_conv(x_lin_view)
        return x_recon, mu, logvar

# Loss function
def VAE_loss(x, x_recon, mu, logvar, beta=1):
    recon_loss = F.binary_cross_entropy(x_recon, x.view(x.size(0), -1), reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld_loss

# Training
latent_dim = 50
vae = VAE(latent_dim).to(device)
optimizer = optim.Adam(vae.parameters())

for epoch in range(10):  # number of epochs
    vae.train()
    train_loss = 0
    for x in train_loader:
        x = x.view(x.size(0), -1)  # Flatten the images
        x = x.to(device)
        x_recon, mu, logvar = vae(x)
        loss = VAE_loss(x, x_recon, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {train_loss / len(train_loader.dataset)}")

# Save the model
torch.save(vae.state_dict(), '/Users/jadeja19/Documents/Hiwi iktd/IKTD ML PROJECT/vae_model.pth')

# Generate new data
vae.eval()
with torch.no_grad():
    z = torch.randn(64, latent_dim).to(device)
    generated_images = vae.decoder(z).cpu()

# Visualize some generated images
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(generated_images[i].view(256, 256), cmap='gray')
    plt.axis('off')
plt.show()
