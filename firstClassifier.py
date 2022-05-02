import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, io
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL 
import torchvision

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128
epochs = 10
learning_rate = 0.1

def main():
    #setting parameters of network
    
    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    print(model)
    #SETTING UP DATA##################################
    # Download training data from open datasets.
    training_data = datasets.MNIST(
        root="./data/",
        train=True,
        download=False,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.MNIST(
        root="./data/",
        train=False,
        download=False,
        transform=ToTensor(),
    )

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    ###################################################
    
    #training network#############################################
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
    ##############################################################
    
    test(test_dataloader, model, loss_fn, device)
    imagepred = test_jpg("./MNIST_JPGS/testSample/img_1.jpg", model, device)
    img = mpimg.imread("./MNIST_JPGS/testSample/img_1.jpg")
    imgplot = plt.imshow(img)
    plt.show()
    print(imagepred)
        
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            #nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits


def train(dataloader, model, loss_fn, optimizer, device):
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

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    
def test_jpg(image_name, model, device):
    
    image = PIL.Image.open(image_name)
    loader = Compose([torchvision.transforms.PILToTensor()])
    image = loader(image).float().unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(image)
    return pred.argmax().item()

if __name__ == "__main__":
    main()
