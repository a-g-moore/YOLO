import torch
from model import Yolo
from data import VOCDataset, Compose
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from PIL import Image

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
            if dat[0,row,col,20] > 0.5:
                x, y, w, h = dat[0,row,col,21:25].tolist()
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


for i in range(10, 10):
    boxes = []
    with open(f"data/labels/0000{i}.txt") as FILE:
        for label in FILE.readlines():
            classLabel, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                    ]
            boxes.append([x, y, width, height, labels[classLabel]])

    print(boxes)
    image = Image.open(f"data/images/0000{i}.jpg")
    plot_image(image, boxes, boxes)

model = Yolo()
model.load_state_dict(torch.load('checkpoint.pth.tar')['state_dict'])
model.to(DEVICE)

puller = iter(dataset)
for i in range(1):
  fig, axs = plt.subplots(3,3,figsize=(10, 10))
  for row in range(3):
    for col in range(3):
      ax = axs[row, col]

      image, label = next(puller)
      image = image.to(DEVICE)
      y = model(image.unsqueeze(0)).reshape((1,7,7,25))
      label = label.unsqueeze(0)
      image = image.to("cpu")
      y = y.to("cpu")
      label = label.to("cpu")

      im = np.array(image.permute(1,2,0))

      predboxes = decodeBoxes(y)
      truthboxes = decodeBoxes(label)
      ax.imshow(im)
      plot_boxes(predboxes, ax, "r", im.shape)
      plot_boxes(truthboxes, ax, "b", im.shape)

  plt.savefig('testoverfit.png')

