# YOLO Version 1 Implementation

This is a PyTorch-based implementation of the object detection algorithm YOLOv1, based on the paper _You Only Look Once: Unified, Real-Time Object Detection_ and the guide by Aladdin Persson.

## Installation

The program expects the Pascal VOC dataset prepared by Aladdin to be in a folder called `data/`. To obtain the data, run the following command using a Kaggle API key to download it from Aladdin's repository:

```sh
kaggle datasets download -d aladdinpersson/pascal-voc-yolo-works-with-albumentations
```

For pre-training on ImageNet, we are currently using the ImageNet 1000 Mini training set with data augmentation. The dataset is available via the following Kaggle API command:

```sh
kaggle datasets download -d ifigotin/imagenetmini-1000
```

To set up the proper environment and install dependencies, run the following commands (assuming that you are running MacOS or Linux):

```sh
python -m venv env
source env/bin/activate
python -m pip install torch torchvision matplotlib pandas tqdm click GPUtil
```

## Training Instructions

Unlike Aladdin's guide, this implementation is intended to do the pre-training on ImageNet described in the paper. Therefore, there are two models available to train, `yolo` and `imagenet`. To train the model, specify the training parameters in `yolo.json` and `imagenet.json`, and then run the following commands:

```sh
python train.py --model imagenet --new
python train.py --model yolo --new --features imagenet
```

The `--features` option loads the feature detection block from a checkpoint of the specified model. For determininistic behavior, you can specify a seed with `--seed`. Omitting `--new` will load a model from a checkpoint: for instance, `yolo` will be loaded from `yolo.pth.tar`. You can also specify a different parameter JSON file with `--params`. For details regarding usage of the training script, run `python train.py --help`. 

## Customization

The architecture of the feature detection networks and both classifiers are stored in `architecture.json`. All networks are built with 'Lazy' layers, so you do not have to worry about input sizes when modifying the architecture. To change the architecture, simply modify the JSON file and re-train the network.

## Other Tools

To visualize the output of the network on some examples, run `visualize.py`. If you get an error saying that CUDA has run out of memory try running `freeGPU.py` to clear CUDA memory. 

# Current Coding Objectives 

- Add testing for both models
- Proper visualization & validation code (non-max suppression, mean average precision)

# Details

Most of the implementation is a direct copy of the paper. The paper was not specific about the design of the ImageNet classifier, so I chose the following simple system: 
- 2x2 average pooling
- 4096 fully connected neurons
- LeakyReLU with slope 0.1
- Dropout with probability 0.5
- 1000 fully connected neurons (output)

Following Aladdin's implementation, we have added batch normalization to each convolutional layer. We have also chosen to use an adaptive learning rate scheduler, Torch's `ReduceLROnPlateau` with default settings. This is the only scheduler for pre-training. The paper does not give exact details about the design of the burn in scheduler for the YOLO training. It is certainly true that raising the learning rate too fast early in training causes the gradients to diverge. I found that the initial learning rate of 1e-3 was also too high. Instead, I opted to linearly interpolate from 1e-4 to 1e-3 over the first 10 epochs, then from 1e-3 to 1e-2 over the next 10 epochs, before switching to the landmarking schedule described in the paper. 

Over the course of getting all the little moving pieces right, I noticed that the model is very sensitive to the relative weighting of the components of the loss function. If width and height are mistakenly normalized with respect to the cell bounding boxes rather than the overall image, for example, the model will not sufficiently emphasize classification of objects, resulting in mode collapse---I found over several failed attempts that it would always classify almost every object as a person. 

## Training Report

A friend provided the machine for training the model. The graphics card was an RTX 3090. The model was pre-trained with the ImageNet classifier for 400 epochs (~6 hours), with an initial average cross-entropy loss of 7.78 (after the first epoch) and a final loss of 0.164. The initial learning rate was chosen to be 10^-1 and the final learning rate was observed to be 10^-3. The validation accuracy after training was 11%, with accuracy on the training set at 91%, indicating a high degree of overfitting. This is obviously nowhere near as good as the 88% test validation reported in the paper, but on the other hand we did not have the resources to train on the full ImageNet-1000 set for 2 weeks. The YOLO model was then trained for 135 epochs following the learning rate schedule described in the paper with ther recommended batch size of 64, with an initial average MSE loss of 0.183 (after the first epoch) and a final loss of 0.0111. However, the results were less than pleasing. Training for 340 epochs using 