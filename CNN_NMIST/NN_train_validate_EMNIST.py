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


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", DEVICE)

#PATH = 'C:/Users/Robocode/Desktop/Sieci-Neuronowe'
PATH = sys.path[0]
filename = 'NN_MNIST.pth'
PATH = os.path.join(PATH, filename)

# Image parameters
n_classes = 10

# Get data
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

# Define initial parameters
learning_rate = 1e-4
batch_size = 50
n_epochs = 30
criterion = nn.CrossEntropyLoss()

# Create network
network = CNN.Network(n_classes, batch_size, n_epochs, learning_rate, criterion).to(DEVICE) # move network to GPU if available

### TRAINING
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
loss_vector_training, loss_vector_test, accuracy_vector_training, accuracy_vector_test = network.train(train_set, test_set, optimizer)
network.save(PATH)

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
plt.show()
plt.savefig(os.path.join(PATH, 'TrainingValidationPlot.png'))

### VISUAL TEST

label_names = (
    '0'
    ,'1'
    ,'2'
    ,'3'
    ,'4'
    ,'5'
    ,'6'
    ,'7'
    ,'8'
    ,'9'
)
fig = plt.figure()
plot_rows = 5
plot_cols = 10
plot_images_data_loader = torch.utils.data.DataLoader(test_set, batch_size = 1, shuffle = True)
for index in range(1, plot_rows * plot_cols + 1):
        plt.subplot(plot_rows, plot_cols, index)
        images, labels = next(iter(plot_images_data_loader)) #Get batch from data loader
        prediction = int(torch.argmax(network(images)).item())
        plt.imshow(np.transpose(images[0].cpu().detach().numpy(), (1, 2, 0)), cmap='gray_r')
        text1 = str(label_names[labels])
        text2 = str(label_names[prediction])
        plt.title(str('T: '+ text1 + '\nP: '+ text2), fontsize=7)
        plt.axis('off')
fig.suptitle('Neural network predictions')
plt.show()