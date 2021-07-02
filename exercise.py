import torch, torchvision #
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_dataset = torchvision.datasets.MNIST(root="./", train=True, download=True, transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,)) ]))
test_dataset = torchvision.datasets.MNIST(root="./", train=False, download=True, transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,)) ]))

