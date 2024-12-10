import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
import torch.nn.functional as F
import os
from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
image_size = 256  # Adjust to your image size
h_dim = 400
z_dim = 20
learning_rate = 1e-3
batch_size = 32
num_epochs = 30
reconstructed_image_dir = '/Users/jadeja19/Documents/Hiwi iktd/data set/allgemein1-5/vaeimages'  # Directory for saving reconstructed images

# Create the directory if it doesn't exist
if not os.path.exists(reconstructed_image_dir):
    os.makedirs(reconstructed_image_dir)


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.images = [img for img in os.listdir(data_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]  # Filter for image files only
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.images[idx])
        image = Image.open(img_name).convert('L')
        if self.transform:
            image = self.transform(image)
        return image



# Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

data_dir = '/Users/jadeja19/Documents/Hiwi iktd/data set/allgemein1-5/01Schraubedrawcorrected'
dataset = CustomDataset(data_dir, transform=transform)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


# ConvVAE model
class ConvVAE(nn.Module):
    def __init__(self, image_channels=1, h_dim=400, z_dim=20):
        super(ConvVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc1 = nn.Linear(32 * (image_size // 4) * (image_size // 4), z_dim)
        self.fc2 = nn.Linear(32 * (image_size // 4) * (image_size // 4), z_dim)
        self.fc3 = nn.Linear(z_dim, 32 * (image_size // 4) * (image_size // 4))

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (32, image_size // 4, image_size // 4)),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        z = self.fc3(z)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


# Loss function
def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


# Model, optimizer
model = ConvVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for i, x in enumerate(data_loader):
        x = x.to(device)
        x_recon, mu, logvar = model(x)
        loss = vae_loss(x_recon, x, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(data_loader.dataset):.4f}')

    # Save the reconstructed images
    if epoch % 10 == 0:
        save_image(x_recon.cpu(), f'/Users/jadeja19/Documents/Hiwi iktd/data set/allgemein1-5/vaeimages.png')

# Save the model
torch.save(model.state_dict(), '/Users/jadeja19/Documents/Hiwi iktd/IKTD ML PROJECT/vae_model.pth')
