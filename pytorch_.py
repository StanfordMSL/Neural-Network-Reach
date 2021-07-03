import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import numpy


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, X, Y):
        'Initialization'
        self.X = X
        self.Y = Y

  def __len__(self):
        'Denotes the total number of samples'
        return X.shape[0]

  def __getitem__(self, index):
        'Generates one sample of data'
        x = torch.tensor(X[index, :], dtype=torch.float32).to(device)
        y = torch.tensor(Y[index, :], dtype=torch.float32).to(device)
        return x, y


class FFReLUNet(nn.Module):
    """
    Implements a feed forward neural network that uses
    ReLU activations for all hidden layers with no activation on the output layer.
    """

    def __init__(self, shape):
        """Constructor for network.
        Args:
            shape (list of ints): list of network layer shapes, which
            includes the input and output layers.
        """
        super(FFReLUNet, self).__init__()
        self.shape = shape
        self.flatten = nn.Flatten()

        # Build up the layers
        layers = []
        for i in range(len(shape) - 1):
            layers.append(nn.Linear(shape[i], shape[i + 1]))
            if i != (len(shape) - 2):
                layers.append(nn.ReLU(inplace=True))

        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass on the input through the network.
        Args:
            x (torch.Tensor): Input tensor dims [batch, self.shape[0]]
        Returns:
            torch.Tensor: Output of network. [batch, self.shape[-1]]
        """
        x = self.flatten(x)
        return self.seq(x)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")






if torch.cuda.is_available(): device = torch.device("cuda")
else:                         device = torch.device("cpu")

params = {}

# import data, split, and construct dataset classes
dataset = "vanderPol"
X = np.load(dataset + "/data/X.npy")
Y = np.load(dataset + "/data/Y.npy")

# normalize data to be zero mean and unit std_dev
X_mean, X_std = torch.mean(X, dim=-2), torch.std(X, dim=-2)
Y_mean, Y_std = torch.mean(Y, dim=-2), torch.std(Y, dim=-2)
params["X_mean"], params["X_std"] = X_mean, X_std
params["Y_mean"], params["Y_std"] = Y_mean, Y_std

X = (X - X_mean) / X_std
Y = (Y - Y_mean) / Y_std

in_dim, out_dim, N = X.shape[1], Y.shape[1], X.shape[0]
print("Nonlinear regression for input dim = " + in_dim + ", output dim = " + out_dim + ", with " + N + " samples.")

split = int(0.90 * N)
training_data = Dataset(X[:split, :], Y[:split, :])
testing_data = Dataset( X[split:, :], Y[split:, :])


# Create data loaders.
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

for x, y in test_dataloader:
    print("\nShape of X_test: ", x.shape)
    print("Shape of Y_test: ", y.shape, y.dtype)
    break


print("Using {} device".format(device))

model = FFReLUNet([28*28, 20, 20, 10]).to(device)

print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


# Export weights
weights = []
for name, param in model.named_parameters():
    print('name: ', name)
    print(type(param))
    print('param.shape: ', param.shape)
    weights.append(param.detach().numpy())
    print('=====')

# save normalization parameters too
np.savez(dataset + "/savedvars", *weights)










# OLD STUFF
# Download training data from open datasets.
# training_data = datasets.FashionMNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor(),
# )

# # Download test data from open datasets.
# test_data = datasets.FashionMNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor(),
# )

# # Define model
# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(28*28, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10),
#             nn.ReLU()
#         )

#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits