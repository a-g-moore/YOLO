import torch
import torch.nn as nn
import json

# Class to package a convolution block, including batch norm
class ConvolutionBlock(nn.Module):
    def __init__(self, outputChannels, activation = nn.LeakyReLU(0.1), **kwargs):
        super(ConvolutionBlock, self).__init__()
        self.block = nn.LazyConv2d(outputChannels, bias = False, **kwargs)
        self.batchnorm = nn.LazyBatchNorm2d()
        self.activation = activation

    def forward(self, inputData):
        return self.activation(self.batchnorm(self.block(inputData)))

class LinearBlock(nn.Module):
    def __init__(self, output_size, activation):
        super(LinearBlock, self).__init__()
        self.block = nn.LazyLinear(output_size)
        self.activation = nn.LeakyReLU(0.1) if activation == "LeakyReLU" else nn.LeakyReLU(1)

    def forward(self, inputData):
        return self.activation(self.block(inputData))

class Model(nn.Module):
    def __init__(self, classifier_name, featureDetector = None, **kwargs):
        super(Model, self).__init__()

        with open("config/architecture.json", "r") as FILE:
            architectures = json.load(FILE)

        self.featureDetector = featureDetector if featureDetector else createNetwork(architectures['feature_detector'])
        self.classifier = createNetwork(architectures[classifier_name])

    def forward(self, inputData):
        return self.classifier(self.featureDetector(inputData))

def createLayer(description):
    if description['type'] == "maxpool":
        return  nn.MaxPool2d(
                kernel_size = description['kernel'],
                stride = description['stride'],
                padding = 0
                )

    if description['type'] == "avgpool":
        return  nn.AvgPool2d(
                kernel_size = description['kernel'],
                stride = description['stride'],
                padding = 0
                )

    if description['type'] == "conv":
        return  ConvolutionBlock(
                description['outputChannels'],
                kernel_size = description['kernel'],
                stride = description['stride'],
                padding = int(description['kernel'] / 2)
                )
    
    if description['type'] == "linear":
        return  LinearBlock(
                description['output_size'],
                description['activation']
                )

    if description['type'] == "flatten":
        return nn.Flatten()

    print(f"Error: Cannot create layer type '{description['type']}'")

def createNetwork(architecture):
    return nn.Sequential(*[createLayer(description) for description in architecture])
