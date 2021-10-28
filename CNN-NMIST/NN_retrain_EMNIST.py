import torch
import torchvision
import torchvision.transforms as transforms
 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
 
torch.set_grad_enabled(True)
torch.set_printoptions(linewidth =120)
 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
 
import os
import sys

import ClassNeuralNetwork as CNN
import ClassIMGDatabase as MyDatabase

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", DEVICE)

#PATH = 'C:/Users/Robocode/Desktop/Sieci-Neuronowe'
PATH = sys.path[0]
filename = 'NN_MNIST.pth'

# Image parameters
n_classes = 10

# Get data
GetCustomDataset = True
if GetCustomDataset == False:
    print("Importing MNIST data")
    train_set = torchvision.datasets.MNIST(
        root='./data/MNIST'
        ,train=True
        ,download=True
        ,transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )
    test_set = torchvision.datasets.MNIST(
        root='./data/MNIST'
        ,train=False
        ,transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )
    print("MNIST data imported")
else:
    PATH_TRAIN = r'D:/DATABASE/TRAIN/'
    PATH_TEST = r'D:/DATABASE/TEST/'
    print("Importing custom data")
    train_set = MyDatabase.ImageDatasetTrain(PATH_TRAIN)
    test_set = MyDatabase.ImageDatasetTest(PATH_TEST)
    print("Custom data imported")

# Define initial parameters
learning_rate = 1e-4
batch_size = 50
n_epochs = 30
criterion = nn.CrossEntropyLoss()

#Create and load network
network = CNN.Network(n_classes).to(DEVICE) # move network to GPU if available
network.load_state_dict(torch.load(os.path.join(PATH, filename)))
network.eval()


# Create loaders
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1, shuffle = True)
train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle = True) # Data

### TRAINING
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
network, loss_vector_training, loss_vector_test, accuracy_vector_training, accuracy_vector_test = CNN.train(
    network, train_set, train_loader, test_set, test_loader, optimizer, n_epochs, learning_rate, criterion
)
network.save(os.path.join(PATH, filename))

x = range(0, int(n_epochs))

ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.xlabel("Epochs")
plt.ylabel("Normalized loss // Accuracy")
plt.plot(x,loss_vector_training/np.max(loss_vector_training), label = "Normalized loss - training")
plt.plot(x,loss_vector_test/np.max(loss_vector_test), label = "Normalized loss - validating")
plt.plot(x,accuracy_vector_training, label = "Accuracy - training")
plt.plot(x,accuracy_vector_test, label = "Accuracy - validating")
plt.legend()
plt.savefig(os.path.join(PATH, 'ReTrainingValidationPlot.png'))
plt.show()