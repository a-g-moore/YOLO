import torch
from model import Model
from data import VOCDataset, Compose
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from PIL import Image
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"
labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

def plot_boxes(boxes, axes, color, shape):
    height, width, _ = shape
    for x, y, w, h, name in boxes:
        cornerX = (x-w/2)
        cornerY = (y-h/2)
        rect = patches.Rectangle(
            (
                cornerX * width, 
                cornerY * height
            ),
            w * width,
            h * height,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        # Add the patch to the Axes
        axes.add_patch(rect)
        axes.text(cornerX * width, cornerY * height - 10, name, fontsize = 14, color=color)

def decodeBoxes(dat):
    boxes = []
    for row in range(7):
        for col in range(7):
            if dat.size()[3] == 25 or dat[0,row,col,20] > dat[0, row, col, 25]:
                i = 0
            else:
                i = 1
            
            if dat[0,row,col,20 + i*5] > 0.5:
                x, y, w, h = dat[0,row,col,(20 + i*5 + 1):(20 + (i+1)*5)].tolist()
                x = (x + col)/7
                y = (y + row)/7
                w = w/7
                h = h/7
                label = labels[torch.argmax(dat[0,row,col,0:20])]

                boxes.append([x,y,w,h,label])

    return boxes

dataset = VOCDataset(
            "data/100examples.csv",
            transform = Compose([
                transforms.Resize((448, 448)), 
                transforms.ToTensor()
                ]),
            img_dir = IMG_DIR,
            label_dir = LABEL_DIR
            )

model = Model(classifier_name = f"yolo_classifier")
model.load_state_dict(torch.load('checkpoints/yolo.pth.tar')['model'])
model.to(DEVICE)

puller = iter(dataset)
for i in range(random.randint(0,50)):
    next(puller)

for i in range(1):
  fig, axs = plt.subplots(3,3,figsize=(10, 10))
  for row in range(3):
    for col in range(3):
      ax = axs[row, col]

      image, label = next(puller)
      image = image.to(DEVICE)
      y = model(image.unsqueeze(0)).reshape((1,7,7,30))
      label = label.unsqueeze(0)
      image = image.to("cpu")
      y = y.to("cpu")
      label = label.to("cpu")

      im = np.array(image.permute(1,2,0))

      truthboxes = decodeBoxes(label)
      ax.imshow(im)
      plot_boxes(decodeBoxes(y), ax, "r", im.shape)
      plot_boxes(truthboxes, ax, "b", im.shape)

  plt.savefig('testoverfit.png')

