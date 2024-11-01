import torch
from torch import nn
from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.transforms import ToTensor, Lambda, Compose
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

        if batch % 20 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0.
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")








# Choose dynamics: "vanderpol", "mpc", "taxinet_dyn", "pend_ctrl"
dynamics = "taxinet_dyn"

if torch.cuda.is_available(): device = torch.device("cuda")
else:                         device = torch.device("cpu")

# import data, normalize, split, and construct dataset classes
# X = numpy.load("models/taxinet/Y_image.npy")
# Y = numpy.load("models/taxinet/X_image.npy")

X = numpy.load("models/taxinet/X_dynamics_1hz.npy")
Y = numpy.load("models/taxinet/Y_dynamics_1hz.npy")

# X = numpy.load("models/Pendulum/X_controlled.npy")
# Y = numpy.load("models/Pendulum/Y_controlled.npy")

# generate quadratic function data
# Define the grid range and step size
# grid_size = 100  # You can adjust this for more or fewer points
# x_values = numpy.linspace(-1, 1, grid_size)
# y_values = numpy.linspace(-1, 1, grid_size)
# x1, x2 = numpy.meshgrid(x_values, y_values)
# X = numpy.vstack([x1.ravel(), x2.ravel()]).T
# Y = (X[:, 0]**2 + X[:, 1]**2).reshape(-1, 1)

X_mean, X_std = numpy.mean(X, axis=0), numpy.std(X, axis=0)
Y_mean, Y_std = numpy.mean(Y, axis=0), numpy.std(Y, axis=0)
X = (X - X_mean) / X_std
Y = (Y - Y_mean) / Y_std

in_dim, out_dim, N = X.shape[1], Y.shape[1], X.shape[0]
split = int(0.90 * N)
training_data = Dataset(X[:split, :], Y[:split, :])
testing_data = Dataset( X[split:, :], Y[split:, :])

print("\n\nNonlinear regression for input dim = " + str(in_dim) + ", output dim = " + str(out_dim) + ", with " + str(split) + " samples.")
print("Using {} device".format(device))


# Create data loaders.
batch_size = 100
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

layer_sizes = numpy.array([in_dim, 16, out_dim])
model = FFReLUNet(layer_sizes).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-6)
print("\n", model)

# Train
epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


# Export weights
weights = []
for name, param in model.named_parameters():
    # print('name: ', name)
    # print(type(param))
    # print('param.shape: ', param.shape)
    weights.append(param.detach().cpu().numpy())
    # print('=====')

    


# save weights and normalization parameters
numpy.savez("models/taxinet/weights_dynamics_1hz_2nd.npz", *weights)
numpy.savez("models/taxinet/norm_params_dynamics_1hz_2nd.npz", X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std, layer_sizes=layer_sizes)

# numpy.savez("models/Pendulum/weights_controlled.npz", *weights)
# numpy.savez("models/Pendulum/norm_params_controlled.npz", X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std, layer_sizes=layer_sizes)


# numpy.savez("models/" + dynamics + "/weights.npz", *weights)
# numpy.savez("models/" + dynamics + "/norm_params.npz", X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std, layer_sizes=layer_sizes)

# numpy.savez("models/quadratic/weights.npz", *weights)
# numpy.savez("models/quadratic/norm_params.npz", X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std, layer_sizes=layer_sizes)