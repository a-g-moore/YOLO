# YOLO Version 1 Implementation

This is a PyTorch-based implementation of the object detection algorithm YOLOv1, based on the paper _You Only Look Once: Unified, Real-Time Object Detection_ and the guide by Aladdin Persson.

## Installation

The program expects the Pascal VOC dataset prepared my Aladdin to be in a folder called `data/`. To obtain the data, run the follow command using a Kaggle API key to download it from Aladdin's repository:

```sh
kaggle datasets download -d aladdinpersson/pascal-voc-yolo-works-with-albumentations
```

To set up the proper environment and install dependencies, run the following commands (assuming that you are running MacOS or Linux):

```sh
python -m venv env
source env/bin/activate
python -m pip install torch torchvision matplotlib pandas tqdm click GPUtil
```

## Training Instructions

Unlike Aladdin's guide, this implementation is intened to do the pre-training on ImageNet described in the paper. Therefore, there are two models available to train, `yolo` and `imagenet`. To train the model, specify the training parameters in `yolo.json` and `imagenet.json`, and then run the following commands:

```sh
python train.py --model imagenet --new
python train.py --model yolo --new --features imagenet
```

The `--features` option loads the feature detection block from a checkpoint of the specified model. For determininistic behavior, you can specify a seed with `--seed`. Omitting `--new` will load a model from a checkpoint: for instance, `yolo` will be loaded from `yolo.pth.tar`. You can also specify a different parameter JSON file with `--params`. For details regarding usage of the training script, run `python train.py --help`. 

## Customization

The architecture of the feature detection networks and both classifiers are stored in `architecture.json`. All networks are built with 'Lazy' layers, so you do not have to worry about input sizes when modifying the architecture. To change the architecture, simply modify the JSON file and re-train the network.

## Other Tools

To visualize the output of the network on some examples, run `visualize.py`. If you get an error saying that CUDA has run out of memory try running `freeGPU.py` to clear CUDA memory. 
