# test.py

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import PetCNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

test_data = datasets.OxfordIIITPet(
    root="data",
    split="test",
    target_types="category",
    download=True,
    transform=transform
)

test_loader = DataLoader(test_data, batch_size=32)

model = PetCNN(num_classes=37).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

print(f"Test accuracy: {100 * correct / total:.2f}%")
