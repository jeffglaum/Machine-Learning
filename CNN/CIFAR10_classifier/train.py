import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from classifier.net import Net
import torch.nn as nn
import torch.optim as optim
import ssl

# Without this line data set downloads will encounter an SSL error.
ssl._create_default_https_context = ssl._create_unverified_context

# Find the best hardware accelerator to use (TODO: mps on the mac is slower than cpu).
#device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Shrink the CIFAR10 images to half size to save processing time.
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 4

# Find or fetch the CIFAR10 training set.
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

# CIFAR10 image set classes.
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Construct the network and optimizer.
classifier_net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(classifier_net.parameters(), lr=0.001, momentum=0.9)

# Train over 10 epocs.
for epoch in range(10):

    running_loss = 0.0

    # Enumerate over the training data in batches.
    for i, (inputs, labels) in enumerate(trainloader, 0):

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = classifier_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[epoch {epoch + 1}, iteration {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished training')

# Save the trained model.
PATH = './cifar_net.pth'
torch.save(classifier_net.state_dict(), PATH)
print('Trained model saved')
