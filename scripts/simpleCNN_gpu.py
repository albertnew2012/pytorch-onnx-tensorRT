import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np


# Transformations for the image
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Downloading the training data
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)

# Loader for the training data
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=2)

# Downloading the test data
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)

# Loader for the test data
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False, num_workers=2)

# Classes in the MNIST dataset
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # 1 input channel, 6 output channels, 5x5 square convolution
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling over a (2, 2) window, stride = 2
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 4x4 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Check if CUDA is available and set the device to GPU if it is, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

net = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # Print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')



correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


# Saving the model
model_path = './simple_cnn.pth'
torch.save(net.state_dict(), model_path)



###################################################################

# Set the model to evaluation mode
net.eval()

# Assuming MNIST is 1 channel, 28x28 images
x = torch.randn(1, 1, 28, 28, requires_grad=True).to(device)

# Export the model
torch.onnx.export(net,               # Model being run
                  x,                 # Model input (or a tuple for multiple inputs)
                  "simple_cnn.onnx", # Where to save the model
                  export_params=True,        # Store the trained parameter weights inside the model file
                  opset_version=10,          # The ONNX version to export the model to
                  do_constant_folding=True,  # Whether to execute constant folding for optimization
                  input_names = ['input'],   # The model's input names
                  output_names = ['output'], # The model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # Variable length axes
                                'output' : {0 : 'batch_size'}})




import onnx

# Load the ONNX model
model = onnx.load("simple_cnn.onnx")

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))




######################################################

# Function to show an image
def imshow(img):
    # img = img / 2 + 0.5  # unnormalize if you used normalization in your transforms
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)  # Correct usage in Python 3


# Show images in a grid
imshow(torchvision.utils.make_grid(images))
img = torchvision.utils.make_grid(images)

plt.imsave('sample.png', images[0][0], cmap='gray')  # Choose a colormap (e.g., 'gray')
