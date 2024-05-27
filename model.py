import torch
import torch.nn as nn
import torchvision.models as models


class ResNetForImageProcessing(nn.Module):
    def __init__(self, output_size=19):
        super(ResNetForImageProcessing, self).__init__()
        # Load pre-trained ResNet model
        self.resnet = models.resnet50(pretrained=True)

        # Modify the first convolutional layer to accept 4 input channels (RGB + mask)
        self.resnet.conv1 = nn.Conv2d(
            4, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Replace the final fully connected layer to output `output_size` parameters
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, output_size)

    def forward(self, x):
        return self.resnet(x)
