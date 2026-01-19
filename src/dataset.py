import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    mean = 0.0
    std = 0.0
    total_images = 0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples
    mean /= total_images
    std /= total_images
    return mean, std

mean=[0.5074, 0.5074, 0.5074]
std=[0.2121, 0.2121, 0.2121]

transform=transforms.Compose([
    transforms.Resize((48, 48)),  # Images are 48x48, resize to maintain consistency
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean, std)
])

train_set=datasets.ImageFolder("dataset/emotions/train/",transform=transform)

train_loader = DataLoader(train_set, batch_size=48, shuffle=True, num_workers=4)