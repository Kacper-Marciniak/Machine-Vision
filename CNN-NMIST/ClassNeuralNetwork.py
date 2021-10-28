import torch
 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
 
torch.set_grad_enabled(True)
torch.set_printoptions(linewidth = 120)

from torch.utils.data import Dataset, DataLoader
 
import numpy as np

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

class Network(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=n_classes) # Output - number of classes
        
    def forward(self, t):
        # Input
        t = t
        
        # Conv1
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        # Conv2
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        # FC1 (linear)
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)
        
        # FC2 (linear)
        t = self.fc2(t)
        t = F.relu(t)
        
        # Output
        t = self.out(t)
        t = F.softmax(t, dim=1)
        
        return t

    def save(self, PATH):
        torch.save(self.state_dict(), PATH)

def train(network, train_set, train_loader, test_set, test_loader, optimizer, n_epochs, learning_rate, criterion):
        
        print("Starting training sequence")
        optimizer= torch.optim.Adam(network.parameters(), lr=learning_rate)

        loss_vector_training = np.zeros(n_epochs)
        loss_vector_test = np.zeros(n_epochs)
        accuracy_vector_training = np.zeros(n_epochs)
        accuracy_vector_test = np.zeros(n_epochs)

        for epoch in range(n_epochs):
            
            total_loss = 0
            total_correct = 0
            
            for batch in train_loader: # Get Batch
                images, labels = batch 
        
                preds = network(images) # Pass Batch
                loss = criterion(preds, labels) # Calculate Loss
        
                optimizer.zero_grad() # Clear grad buffor
                loss.backward() # Calculate Gradients
                optimizer.step() # Update Weights

                total_correct += get_num_correct(preds, labels)
                total_loss += loss.item()
        
            print(
                "_"*25+"\n",
                "epoch", epoch, 
                "\n\tTraining total loss:", total_loss,
                "\tTraining accuracy:", total_correct / len(train_set)
            )
            
            total_loss_test, total_correct_test = test(network, test_set, test_loader, criterion)

            loss_vector_training[epoch] = float(total_loss)
            loss_vector_test[epoch] = float(total_loss_test)
            accuracy_vector_training[epoch] = float(total_correct) / len(train_set)
            accuracy_vector_test[epoch] = float(total_correct_test) / len(test_set)

            if epoch > 0: print( 
                "\n\tTraining  loss diff.:", float(loss_vector_training[epoch]-loss_vector_training[epoch-1]),
                "\tTraining accuracy diff.:", float(accuracy_vector_training[epoch]-accuracy_vector_training[epoch-1]),
                "\n\tValidating  loss diff.:", float(loss_vector_test[epoch]-loss_vector_test[epoch-1]),
                "\tValidating accuracy diff.:", float(accuracy_vector_test[epoch]-accuracy_vector_test[epoch-1])
            )
        
        return network, loss_vector_training, loss_vector_test, accuracy_vector_training, accuracy_vector_test


def test(network, test_set, test_loader, criterion):
        
        total_loss = 0
        total_correct = 0
       
        for batch in test_loader:
            images, labels = batch
            
            preds = network(images)
            
            loss = criterion(preds, labels) # Calculate Loss
            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)
        
        print(
            '\tValidation total loss:', total_loss,
            '\tValidation accuracy:', total_correct/len(test_set)
        )

        return total_loss, total_correct 
