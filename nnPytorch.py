# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


print("OK")

# a to dataset
x_values = [i for i in range(11)]

# convert to numpy
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

print(" x values: ", x_values)
print ("x train after shape: ", x_train)


# fit an equation y = 2x + 1
y_values =[2*i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

print(" y values in 2D: ",y_values)

# create a class
class LinearRegressionModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModule, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


input_dim = 1
output_dim = 1

print(" dimensions: ",input_dim,output_dim)

model = LinearRegressionModule(input_dim, output_dim)

# instantiate lossClass
criterion = nn.MSELoss()

# instantiate optimizer class
learningRate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

# train the model
epochs = 100

for epoch in range(epochs):
    epoch += 1

    # convert numpy array to torch Variable
    inputs = Variable(torch.from_numpy(x_train))

    labels = Variable(torch.from_numpy(y_train))

    # clear gradients w.r.t parameters
    optimizer.zero_grad()

    # Forward to get outputs
    outputs = model(inputs)

    # Calculate loss
    loss = criterion(outputs, labels)

    # Getting gradients w.r.t parameters
    loss.backward()

    # Updating parameters
    optimizer.step()

    print(' epoch {}, loss {}'.format(epoch, loss.item()))


# purely inference
predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
print(" predicted: ", predicted)
print(" train set: ", y_train)

#clear figure
plt.clf()

# get predictions
plt.plot(x_train, y_train, 'go', label = ' True Data ', alpha = 0.5)
plt.legend(loc='best')
plt.show()






#
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     y = torch.ones_like(x, device=device)
#     x = x.to(device)
#     z = x + y
#     print(z)
#     print(z.to("cpu", torch.double))
# else:
#     print("No CUDA available")
