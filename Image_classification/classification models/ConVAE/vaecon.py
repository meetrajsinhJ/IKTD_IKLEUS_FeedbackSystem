import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data import random_split
from torchvision.utils import save_image
import torch.nn.functional as F
import os
from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
image_size = 128  # Adjust to your image size
h_dim = 400
z_dim = 50
learning_rate = 0.001
batch_size = 32
num_epochs = 10
validation_split = 0.2
reconstructed_image_dir = '/Users/jadeja19/Documents/Hiwi iktd/data set/allgemein1-5/vaeimages'  # Directory for saving reconstructed images

# Create the directory if it doesn't exist
if not os.path.exists(reconstructed_image_dir):
    os.makedirs(reconstructed_image_dir)


# Custom dataset
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.images = [img for img in os.listdir(data_dir) if img.lower().endswith(
            ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]  # Filter for image files only
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

data_dir = '/Users/jadeja19/Documents/Hiwi iktd/data set/allgemein1-5/02corrected'
dataset = CustomDataset(data_dir, transform=transform)

# Splitting the dataset into training and validation sets
num_val = int(len(dataset) * validation_split)
num_train = len(dataset) - num_val
train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

# Creating data loaders for training and validation sets
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# ConvVAE model
class VAE(nn.Module):
    def __init__(self, image_channels=1, h_dim=400, z_dim=50):
        super(VAE, self).__init__()
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
model = VAE(image_channels=1, h_dim=h_dim, z_dim=z_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Function to evaluate the model on the validation set
def evaluate_model(model, data_loader):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x in data_loader:
            x = x.to(device)
            x_recon, mu, logvar = model(x)
            loss = vae_loss(x_recon, x, mu, logvar)
            val_loss += loss.item()
    return val_loss / len(data_loader.dataset)

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for i, x in enumerate(train_loader):
        x = x.to(device)
        x_recon, mu, logvar = model(x)
        loss = vae_loss(x_recon, x, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    val_loss = evaluate_model(model, val_loader)
    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss / len(train_loader.dataset):.4f}, Val Loss: {val_loss:.4f}')

    # Save the reconstructed images
    if epoch % 10 == 0:
        save_path = os.path.join(reconstructed_image_dir, f'/Users/jadeja19/Documents/Hiwi iktd/data set/allgemein1-5/re02.png')
        save_image(x_recon.cpu(), save_path)

# Save the model
torch.save(model.state_dict(), '/Users/jadeja19/Documents/Hiwi iktd/IKTD ML PROJECT/vae_model.pth')
