import torch
import pandas as pd
import cv2
import os
import albumentations
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

LABEL_NAMES = [
    'aeroplane', 
    'bicycle', 
    'bird', 
    'boat', 
    'bottle', 
    'bus', 
    'car', 
    'cat', 
    'chair', 
    'cow', 
    'diningtable', 
    'dog', 
    'horse', 
    'motorbike', 
    'person', 
    'pottedplant', 
    'sheep', 
    'sofa', 
    'train', 
    'tvmonitor'
    ]

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, transform, split_size = 7, num_classes = 20, ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.split_size = split_size
        self.num_classes = num_classes

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        label = self._read_label(label_path)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed = self.transform(
            image = image,
            bboxes = label['bboxes'],
            class_ids = label['class_ids']
            )
            
        image = torch.tensor(transformed['image'], dtype=torch.float).permute(2,0,1) / 255
        label = list_to_label(transformed, self.split_size, self.num_classes)

        return image, label
    
    def _read_label(self, path):
        boxes = {
            "bboxes": [],
            "class_ids": []
        }

        with open(path) as f:
            for label in f.readlines():
                class_id, x, y, width, height = [float(x) for x in label.replace("\n", "").split()]

                boxes['bboxes'].append([x, y, width, height])
                boxes['class_ids'].append(int(class_id))

        return boxes

def list_to_label(label, split_size=7, num_classes=20):
    """Transforms a label in list form (as in the filesystem or readable by
    the visualization functions) to the (7,7,25) label tensor suitable for 
    training the network.
    """

    label_matrix = torch.zeros((split_size, split_size, num_classes + 5))
    for box, id in zip(label['bboxes'], label['class_ids']):
        x, y, width, height = box
        row, col  = int(split_size * y), int(split_size * x)

        # Only allow one cell per box. Do not allows boxes whose centers do not lie in a valid cell. 
        if label_matrix[row, col, num_classes] or not torch.all(torch.tensor([row,col]) == torch.clamp(torch.tensor([row,col]), min=0, max=split_size-1)): continue

        # Transform the box to relative coordinates
        relativeX, relativeY = x * split_size - col, y * split_size - row
        #width, height = (width, height * split_size)

        # Enter the label into the tensor
        label_matrix[row, col, num_classes] = 1
        label_matrix[row, col, (num_classes+1):(num_classes+5)] = torch.tensor([relativeX, relativeY, width, height])
        label_matrix[row, col, id] = 1

    return label_matrix


def label_to_list(label, split_size = 7, num_classes = 20, threshold = 0.5):
    """Transforms the labels in (7,7,30) or (7,7,25) tensor format into list
    type labels used in the data files and the visualization functions.
    """

    boxes = {
        "bboxes": [],
        "class_ids": [],
        "confidences": []
    }

    for row in range(7):
        for col in range(7):
            if label.size()[2] == num_classes + 5 or label[row,col,num_classes] > label[row, col, num_classes+5]:
                i = 0
            else:
                i = 1
            
            if label[row,col,num_classes + i*5] > threshold:
                x, y, w, h = label[row,col,(num_classes + i*5 + 1):(num_classes + (i+1)*5)].tolist()
                x = (x + col)/split_size
                y = (y + row)/split_size
                w = w
                h = h
                class_index = torch.argmax(label[row,col,:num_classes])

                boxes['bboxes'].append([x,y,w,h])
                boxes['class_ids'].append(class_index)
                boxes['confidences'].append(label[row,col,num_classes + i*5])

    return boxes

def get_VOC_dataset(params, augment = True):
    transform = albumentations.Compose([
                    #albumentations.Resize(width = 448, height = 448),
                    albumentations.HorizontalFlip(p=0.5),
                    albumentations.RandomResizedCrop(width = 448, height = 448, scale=(0.7, 1.0)),
                    albumentations.ColorJitter(),
                    albumentations.Blur(blur_limit = 7, always_apply = False, p = 0.5),
                    albumentations.GaussNoise(var_limit = 40, mean = 0, per_channel = True, always_apply = False, p=0.5)
                    ], bbox_params=albumentations.BboxParams(format="yolo", min_visibility=0.75, label_fields=['class_ids'])
                    ) if augment else albumentations.Compose([
                    albumentations.Resize(width = 448, height = 448),
                    ], bbox_params=albumentations.BboxParams(format="yolo", min_visibility=0.75, label_fields=['class_ids'])
                    )

    return VOCDataset(
                params['training_csv'],
                transform = transform,
                img_dir = params['img_dir'],
                label_dir = params['label_dir']
                )
            
def get_ImageNet_dataset(params, augment = True):
    transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees = (-30, 30)),
                    transforms.RandomResizedCrop(size = 224, scale = (0.33, 1.0)),
                    transforms.ColorJitter(brightness=0.3, hue = 0.2),
                    transforms.RandomPosterize(bits=4, p = 0.2),
                    transforms.RandomAdjustSharpness(sharpness_factor = 2),
                    transforms.RandomEqualize(),
                    transforms.ToTensor(),
                    ]) if augment else transforms.Compose([
                        transforms.Resize(size = (224, 224)),
                        transforms.ToTensor()
                    ])

    return datasets.ImageFolder(params['data_folder'], transform)