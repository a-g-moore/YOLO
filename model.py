import torch
import torch.nn as nn

# Architecture for the convolutional network
architecture = [
        {"type": "convolution", "kernel": 7, "outputChannels": 64, "stride": 2},
        {"type": "pool", "kernel": 2, "stride": 2},
        {"type": "convolution", "kernel": 3, "outputChannels": 192, "stride": 1},
        {"type": "pool", "kernel": 2, "stride": 2},
        {"type": "convolution", "kernel": 1, "outputChannels": 128, "stride": 1},
        {"type": "convolution", "kernel": 3, "outputChannels": 256, "stride": 1},
        {"type": "convolution", "kernel": 1, "outputChannels": 256, "stride": 1},
        {"type": "convolution", "kernel": 3, "outputChannels": 512, "stride": 1},
        {"type": "pool", "kernel": 2, "stride": 2},
        {"type": "convolution", "kernel": 1, "outputChannels": 256, "stride": 1},
        {"type": "convolution", "kernel": 3, "outputChannels": 512, "stride": 1},
        {"type": "convolution", "kernel": 1, "outputChannels": 256, "stride": 1},
        {"type": "convolution", "kernel": 3, "outputChannels": 512, "stride": 1},
        {"type": "convolution", "kernel": 1, "outputChannels": 256, "stride": 1},
        {"type": "convolution", "kernel": 3, "outputChannels": 512, "stride": 1},
        {"type": "convolution", "kernel": 1, "outputChannels": 256, "stride": 1},
        {"type": "convolution", "kernel": 3, "outputChannels": 512, "stride": 1},
        {"type": "convolution", "kernel": 1, "outputChannels": 256, "stride": 1},
        {"type": "convolution", "kernel": 3, "outputChannels": 512, "stride": 1},
        {"type": "convolution", "kernel": 1, "outputChannels": 512, "stride": 1},
        {"type": "convolution", "kernel": 3, "outputChannels": 1024, "stride": 1},
        {"type": "pool", "kernel": 2, "stride": 2},
        {"type": "convolution", "kernel": 1, "outputChannels": 512, "stride": 1},
        {"type": "convolution", "kernel": 3, "outputChannels": 1024, "stride": 1},
        {"type": "convolution", "kernel": 1, "outputChannels": 512, "stride": 1},
        {"type": "convolution", "kernel": 3, "outputChannels": 1024, "stride": 1},
        {"type": "convolution", "kernel": 3, "outputChannels": 1024, "stride": 1},
        {"type": "convolution", "kernel": 3, "outputChannels": 1024, "stride": 2},
        {"type": "convolution", "kernel": 3, "outputChannels": 1024, "stride": 1},
        {"type": "convolution", "kernel": 3, "outputChannels": 1024, "stride": 1},
        ]

# Class to package a convolution block, including batch norm
class ConvolutionBlock(nn.Module):
    def __init__(self, inputChannels, outputChannels, activation = nn.ReLU(), **kwargs):
        super(ConvolutionBlock, self).__init__()
        self.block = nn.Conv2d(inputChannels, outputChannels, bias = False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(outputChannels)
        self.activation = activation

    def forward(self, inputData):
        return self.activation(self.batchnorm(self.block(inputData)))

class Yolo(nn.Module):
    def __init__(self, architecture = architecture, inputChannels = 3, hiddenSize = 4096, splitSize = 7, convNet = None, **kwargs):
        super(Yolo, self).__init__()
        self.convNet = convNet if convNet else createConvNet(architecture, inputChannels)
        self.fullNet = self._createFullNet(hiddenSize, splitSize)

    def forward(self, inputData):
        return self.fullNet(self.convNet(inputData))

    def _createFullNet(self, hiddenSize, splitSize):
        return nn.Sequential(
                nn.Flatten(),
                nn.Linear(splitSize * splitSize * 1024, hiddenSize),
                nn.Dropout(0.0),
                nn.LeakyReLU(0.1),
                nn.Linear(hiddenSize, splitSize * splitSize * 25)
                )



def createLayer(inputChannels, description):
    if description['type'] == "pool":
        layer = nn.MaxPool2d(
                kernel_size = description['kernel'],
                stride = description['stride'],
                padding = 0
                )
        return (inputChannels, layer)

    if description['type'] == "convolution":
        layer = ConvolutionBlock(
                inputChannels, 
                description['outputChannels'],
                kernel_size = description['kernel'],
                stride = description['stride'],
                padding = int(description['kernel'] / 2)
                )
        return (description['outputChannels'], layer)

    print(f"Error: Cannot create layer type '{description['type']}'")

def createConvNet(architecture, inputChannels):
    layers = []
    currentChannels = inputChannels
    for layerDescription in architecture:
        currentChannels, layer = createLayer(currentChannels, layerDescription)
        layers.append(layer)

    return nn.Sequential(*layers)
