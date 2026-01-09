import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.datasets import mnist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter

train_batch_size = 64   # 训练数据集的批次大小，即每次迭代加载到模型中的训练样本数
test_batch_size = 1000  # 测试数据集的批次大小，即每次迭代加载到模型中的测试样本数
learning_rate = 0.01    # 学习率，控制模型权重更新的步长
num_epochs = 20        # 训练的总轮数，即整个训练数据集将被迭代多少次
lr = 0.01   
momentum = 0.5         # 动量因子，用于加速SGD优化器的收敛速度

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
train_dataset = mnist.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = mnist.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)

class Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.BatchNorm1d(n_hidden_1))
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.BatchNorm1d(n_hidden_2))
        self.out = nn.Sequential(
            nn.Linear(n_hidden_2, out_dim))
        
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.out(x), dim=1)
        return x
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net(28*28, 256, 64, 10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

losses = []
acces = []
eval_losses = []
eval_acces = []
writer = SummaryWriter(log_dir='logs',comment='train-loss')

for epoch in range(num_epochs):
    train_loss = 0
    train_acc = 0
    model.train()
    if epoch%5==0:
        optimizer.param_groups[0]['lr'] *= 0.9
        print("学习率:{:.6f}".format(optimizer.param_groups[0]['lr']))
    for img, label in train_loader:
        img = img.to(device)
        label = label.to(device)
        out = model(img)
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        writer.add_scalar('Train', train_loss/len(train_loader), epoch)
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        train_acc += acc

    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))

    eval_loss = 0
    eval_acc = 0

    model.eval()
    for img, label in test_loader:
        img = img.to(device)
        label = label.to(device)
        img = img.view(img.size(0), -1)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.item()
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        eval_acc += acc

    eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))
    print('epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'
          .format(epoch + 1, num_epochs, train_loss / len(train_loader), train_acc / len(train_loader),
                  eval_loss / len(test_loader), eval_acc / len(test_loader)))
    
    plt.title('Train and Eval Loss')
    plt.plot(np.arange(len(losses)), losses)
    plt.legend(['Train Loss'], loc='upper right')
