import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

np.random.seed(100)
torch.manual_seed(100)

input_size = 1
output_size = 1
learing_rate = 0.01
num_epochs = 100

x_train = np.linspace(-1, 1, 100).reshape(100, 1)
y_train = 3 * np.power(x_train, 2) + 2 + 0.2 * np.random.randn(100, 1)

writer = SummaryWriter(log_dir='logs', comment='linear')

model = nn.Linear(input_size, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learing_rate)

for epoch in range(num_epochs):
    inputs = torch.from_numpy(x_train).float()
    targets = torch.from_numpy(y_train).float()

    outputs = model(inputs)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar('Loss/train', loss.item(), epoch)

    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

writer.close()

print(f'\n训练完成，最终参数')
print(f'权重: {model.weight.item():.4f}, 偏置: {model.bias.item():.4f}')
print(f'真实函数：y = 3x^2 + 2')
print(f'拟合函数：y = {model.weight.item():.4f}x + {model.bias.item():.4f}')
# tensorboard --logdir=logs