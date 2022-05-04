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
batch_size = 64
epochs = 5
learning_rate = 0.01

def main():
    #setting parameters of network
    
    model = ConvNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    
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
        test(test_dataloader, model, loss_fn, device)
        print("-------------------------------\n")
    ##############################################################
    print("Done!")
    
    
    
    jpgpath = ""
    num = 1
    while jpgpath != "exit":
        #jpgpath = input("Please enter a filepath or 'exit' to exit:\n> ")
        if jpgpath == "exit":
            print("Exiting...")
            break
        jpgpath = "./MNIST_JPGS/testSample/img_"+str(num)+".jpg"
        num+=1
        imagepred = test_jpg(jpgpath, model, device)
        print("Classifier",imagepred)
        img = mpimg.imread(jpgpath)
        plt.imshow(img)
        #plt.title("Image classified as",str(imagepred))
        plt.show()
        
        
        
# Define model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = 8,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels = 8,
                out_channels = 16,
                kernel_size = 3,
                padding = 1
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(
                kernel_size = 3,
                padding=1
            ),
            nn.Flatten(),
            nn.Linear(1600, 512),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.ReLU(),
            nn.Linear(64,10)
        )

    def forward(self, x):
        #x = self.flatten(x)
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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} ")

    
def test_jpg(image_name, model, device):
    
    image = PIL.Image.open(image_name)
    loader = Compose([torchvision.transforms.PILToTensor()])
    image = loader(image).float().unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(image)
    return pred.argmax().item()

if __name__ == "__main__":
    main()