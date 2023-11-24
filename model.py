import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F


class WakeWordDetector(pl.LightningModule):
    def __init__(self):
        super(WakeWordDetector, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(-1, 64)
        x = self.fc(x)
        return torch.sigmoid(x)

    def training_step(self, batch):
        x, y = batch
        x = torch.cat(x, dim=0)
        y = torch.stack(y)
        y_hat = self(x)
        print(y)
        loss = F.binary_cross_entropy(y_hat, y.view(-1, 1).type_as(y_hat))
        # loss = sigmoid_focal_loss(y_hat, y.view(-1, 1).type_as(y_hat), reduction="mean")
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        x = torch.cat(x, dim=0)
        y = torch.stack(y)
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y.view(-1, 1).type_as(y_hat))
        # loss = sigmoid_focal_loss(y_hat, y.view(-1, 1).type_as(y_hat), reduction="mean")
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
