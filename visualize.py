import torch
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import albumentations, torchvision
import math
import cv2
from torch.utils.data import DataLoader
from data import get_VOC_dataset
from model import Model
from checkpoint import load_checkpoint
from loss import YoloLoss, intersection_over_union

from data import label_to_list, LABEL_NAMES

def nms_one_class(boxes, threshold=0.2):
    boxes = sorted(boxes, key=lambda box: box["confidence"], reverse=True)

    i = 0
    while i < len(boxes):
        ref_box = boxes[i]
        boxes = list(filter(lambda box: box is ref_box or intersection_over_union(torch.tensor(ref_box['bbox']), torch.tensor(box['bbox'])) < threshold, boxes))
        i+=1

    return boxes

def non_max_suppression(labels):
    separated_classes = list(map(lambda x: [], range(20)))
    for bbox, id, confidence in zip(labels['bboxes'], labels['class_ids'], labels['confidences']):
        separated_classes[id].append({"bbox": bbox, "confidence": float(confidence)})
    
    cleaned_classes = list(map(nms_one_class, separated_classes))

    final = {
        "bboxes": [],
        "class_ids": [],
        "confidences": []
    }
    for i, class_list in enumerate(cleaned_classes):
        for box in class_list:
            final['bboxes'].append(box['bbox'])
            final['class_ids'].append(i)
            final['confidences'].append(box['confidence'])
    
    return final

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

def plot_box(img, bbox, name, color, thickness=2):
    """Visualizes a single bounding box on the image"""

    img = img.astype(np.uint8).copy()
    imagewidth = img.shape[1]
    imageheight = img.shape[0]
    x, y, w, h = bbox
    x *= imagewidth
    y *= imageheight
    w *= imagewidth
    h *= imageheight
    x_min, x_max, y_min, y_max = int(x - w/2), int(x + w/2), int(y - h/2), int(y + h/2)
   
    img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    img = cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
    img = cv2.putText(
        img,
        text=name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=(255,255,255), 
        lineType=cv2.LINE_AA,
    )
    return img

def plot_boxes(image, labels, color = (255, 0, 0)):
    for bbox, id, confidence in zip(labels['bboxes'], labels['class_ids'], labels['confidences']):
        image = plot_box(image, bbox, f"{LABEL_NAMES[id]} {int(100*confidence)}%", color)

    return image

def visualize(device, model, data_loader, plots_per_row = 4):
    
    model.eval()
    
    num_samples = 16
    num_rows = math.ceil(num_samples / plots_per_row)
    fig, axs = plt.subplots(num_rows, plots_per_row, figsize=(4*plots_per_row, 4*num_rows))
    for i in range(num_samples):
        images, targets = next(iter(data_loader))
        inferences = model(images.to(device)).reshape(1, 7, 7, 30).to('cpu')

        #loss = YoloLoss()
        #print(loss(inferences, targets).item())

        ax = axs[int(i/plots_per_row), i % plots_per_row]
        target_labels = label_to_list(targets[0, ...])
        inference_labels = label_to_list(inferences[0, ...], threshold = 0.05)
        inference_labels = non_max_suppression(inference_labels)
        
        
        image = (255 * images[0, ...]).type(torch.ByteTensor).permute(1, 2, 0).numpy()
        
        #image = plot_boxes(image, target_labels, color = (230, 30, 30))
        image = plot_boxes(image, inference_labels, color = (15, 80, 220))
        ax.imshow(image)
    
    plt.savefig('plot.png')

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #VGG = torchvision.models.vgg16(weights = torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES)
    #featureDetector = VGG.features[:-1]
    featureDetector = None
    model = Model(classifier_name = f"yolo_classifier", featureDetector=featureDetector).to(device)

    load_checkpoint(model, None, None, "yolo")

    dataset = get_VOC_dataset(params = {
            'training_csv': "data/test.csv",
            'img_dir': "data/images",
            'label_dir': "data/labels"
        },
        augment=False)

    data_loader = DataLoader(
            dataset = dataset,
            batch_size = 1,
            num_workers = 1,
            pin_memory = True,
            shuffle = True,
            drop_last = True
            )

    visualize(device, model, data_loader)

if __name__ == "__main__":
    main()