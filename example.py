import torch
import sys
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/mnist")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters

input_size = 784 #28X28
hidden_size =  1500
num_classes = 10
num_epoch = 8

batch_size = 100

learning_rate = 0.004

#MNIST
train_dataset = torchvision.datasets.MNIST(root='./data',train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data',train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

examples = iter(train_loader)
print(examples)
samples, labels = next(examples)#updated

print(samples.shape,labels.shape)
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0],cmap="gray")

img_grid = torchvision.utils.make_grid(samples)
writer.add_image('mnist_images',img_grid)
#two hidden layer NeuralNet
class NeuralNet(nn.Module):
    def __init__(self, input_size,hidden_size,num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.l2 = nn.Linear(hidden_size,num_classes)
        self.activation = nn.ReLU()
    def forward(self, x):
        out = self.l1(x) #input -> hidden
        out = self.activation(out) #activation on hidden
        out=self.l2(out)#hidden->output
        
        return out

model = NeuralNet(input_size,hidden_size,num_classes)
model = model.to(device)
# loss optimizer
criterion =nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
writer.add_graph(model, samples.reshape(-1,28*28).to(device))
#training loop

n_total_steps = len(train_loader)
running_loss = 0.0
running_correct = 0.0

for epoch in range(num_epoch):
    for i,(images, labels) in enumerate(train_loader):
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)

        #forwards
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backwards

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predictions = torch.max(outputs, 1)
        running_correct +=(predictions == labels).sum().item()

        if(i+1)%100==0:
            print(f'epoch {epoch+1}/{num_epoch}, step{i+1}/{n_total_steps}, loss= {loss.item():.4f}')
            writer.add_scalar('training loss', running_loss/100, epoch*n_total_steps+i)
            writer.add_scalar('accuracy', running_correct/100, epoch*n_total_steps+i)
            running_loss=0.0
            running_correct=0.0
#test
labels = []
preds = []

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels1 in test_loader:
        images = images.reshape(-1,28*28).to(device)
        labels1 = labels1.to(device)
        outputs = model(images)
        max_pos = 0
        max_val = 0
        for o in outputs:
            if(max_pos == 0):
                max_val=o
            if o>max_val:
                max_pos =+1
        outputs = []

        n_samples += labels1.shape[0]
        n_correct += (predictions == labels1).sum().item()
        class_predictions = [F.softmax(output,dim=0) for output in outputs]
        preds.append(class_predictions)
        labels.append(predictions)
    preds = torch.cat([torch.stack(batch) for batch in preds])
    labels = torch.cat(labels)

acc = 100.0* (n_correct/n_samples)
print(f'accuracy = {acc}')

classes=range(10)
for i in classes:
    labels_i = labels==i
    preds_i = preds[:, i]
    writer.add_pr_curve(str(i),labels_i,preds_i,global_step=0)
writer.close()
