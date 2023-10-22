import torch
import torch.nn as nn

x = torch.randn(10, 3)
y = torch.randn(10, 2)
# Build a fully connected layer.
linear = nn.Linear(3, 2)

# Build loss function and optimizer.
criterion = nn.MSELoss()

print(list(linear.parameters()))
# 优化方法选用随机梯度下降，学习率为0.01
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# Forward pass.
pred = linear(x)

# Compute loss.
loss = criterion(pred, y)
print('loss:', loss.item())

# Backward pass.
loss.backward()
print('dL/dw: ', linear.weight.grad)
print('dL/db: ', linear.bias.grad)

# 1-step gradient descent.
optimizer.step()

# Print out the loss after 1-step gradient descent.
pred = linear(x)
loss = criterion(pred, y)
print(loss.item())

