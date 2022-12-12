import torch
import pandas as pd
from PIL import Image
import os

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, splitSize = 7, numClasses = 20, transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.splitSize = splitSize
        self.numClasses = numClasses

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        # Convert To Cells
        label_matrix = torch.zeros((self.splitSize, self.splitSize, self.numClasses + 5))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            
            row, col  = int(self.splitSize * y), int(self.splitSize * x)
            
            if label_matrix[row, col, 20]:
                continue


            relativeX, relativeY = self.splitSize * x - col, self.splitSize * y - row
            width, height = (width * self.splitSize, height * self.splitSize)

            label_matrix[row, col, 20] = 1
            label_matrix[row, col, 21:25] = torch.tensor([relativeX, relativeY, width, height])
            label_matrix[row, col, class_label] = 1

        #print(boxes)
        #print(decodeBoxes(label_matrix.unsqueeze(0)))
        #print(label_matrix.shape)

        return image, label_matrix
