import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.models import resnet18
from torchvision.transforms import ToTensor, Compose, Normalize

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = resnet18(num_classes=10) 
        self.model.conv1 = nn.Conv2d(1,64,kernel_size=3,padding=1, bias=False)
        
    def forward(self, x):
        return self.model(x)
    

model = Net().to(device)

data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

train_loader = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=True), batch_size=128, shuffle=True)
val_loader = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=False), batch_size=128, shuffle=False)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

val_metrics = {"accuracy": Accuracy(), "loss": Loss(criterion)}

train_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)